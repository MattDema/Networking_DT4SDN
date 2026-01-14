# physical_controller.py
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
import json

# NEW IMPORTS FOR REAL-TIME TOPOLOGY
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link, get_host


class PhysicalTwinController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(PhysicalTwinController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        
        # We still keep the metadata for the "Type" (e.g. "Linear")
        # But switches and links will be overwritten by live discovery
        self.topology_metadata = {"type": "Unknown", "switches": [], "links": [], "hosts": []}

        wsgi = kwargs['wsgi']
        wsgi.register(TopologyController, {'physical_controller': self})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install Table-Miss Flow Entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info(f"Switch {datapath.id} connected. Table-miss installed.")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    # --- Helper methods for REST Controller ---
    def set_topology(self, data):
        self.topology_metadata = data
        self.logger.info(f"Topology metadata updated via API: {data.get('type')}")

    def get_topology(self):
        """
        Return Hybrid Topology:
        - Type: from static JSON
        - Switches/Links (S-S): from Live Ryu Discovery (LLDP)
        - Hosts/Link (H-S): from static JSON (cached)
        """
        
        # 1. Discover Switches Live
        ryu_switches = get_switch(self, None)
        live_switches = [f"s{s.dp.id}" for s in ryu_switches]
        
        # 2. Discover Links Live (S-S only)
        ryu_links = get_link(self, None)
        live_links = []
        for l in ryu_links:
            src = f"s{l.src.dpid}"
            dst = f"s{l.dst.dpid}"
            live_links.append([src, dst])
            
        # 3. Hosts & Host Links (Static Fallback)
        final_hosts = self.topology_metadata.get("hosts", [])
        static_links = self.topology_metadata.get("links", [])
        static_switches_list = self.topology_metadata.get("switches", [])
        
        # Start with live links
        final_links = list(live_links)
        
        # Merge Host Links from Static Metadata
        # Logic: If a link in static metadata involves a node that is NOT a known switch, it's a host link.
        for link in static_links:
            u, v = link[0], link[1]
            
            # Check if u or v is a host (i.e., not in the static switch list)
            # The static switch list is reliable for names like "s1", "s2".
            u_is_switch = (u in static_switches_list)
            v_is_switch = (v in static_switches_list)
            
            # If it's NOT a Switch-to-Switch link, we assume it's a Host link and keep it.
            if not (u_is_switch and v_is_switch):
                final_links.append(link)

        return {
            "type": self.topology_metadata.get("type", "Unknown"),
            "switches": live_switches if live_switches else static_switches_list, # Fallback if live detection empty
            "hosts": final_hosts,
            "links": final_links
        }


class TopologyController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(TopologyController, self).__init__(req, link, data, **config)
        self.physical_controller = data['physical_controller']

    @route('topology', '/topology/metadata', methods=['GET'])
    def get_metadata(self, req, **kwargs):
        """Dashboard calls this to get current topology info"""
        body = json.dumps(self.physical_controller.get_topology())
        # FIX: Specificare charset='utf-8' per evitare l'errore webob
        return Response(content_type='application/json', charset='utf-8', body=body)

    @route('topology', '/topology/metadata', methods=['POST'])
    def set_metadata(self, req, **kwargs):
        """Mininet script calls this to set current topology info"""
        try:
            # Parse JSON body
            data = json.loads(req.body) if req.body else {}
            self.physical_controller.set_topology(data)
            # FIX: Anche qui specificare charset='utf-8'
            return Response(content_type='application/json', charset='utf-8', body=json.dumps({"status": "success"}))
        except Exception as e:
            return Response(status=500, body=str(e))