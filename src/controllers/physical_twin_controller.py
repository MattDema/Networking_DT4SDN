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
        self.topology_metadata = {"type": "Unknown", "switches": [], "links": []}

        wsgi = kwargs['wsgi']
        wsgi.register(TopologyController, {'physical_controller': self})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install Table-Miss Flow Entry (Send unknown packets to Controller)
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
        # --- Standard L2 Learning Switch Logic ---
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
        # Learn the source
        self.mac_to_port[dpid][src] = in_port

        # Decide output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow to avoid Packet-In next time
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
        # We only care about the "type" and maybe hosts/names from the static file
        # The links/switches will be ignored in favor of live discovery
        self.topology_metadata["type"] = data.get("type", "Unknown")
        # We can keep hosts from static config as a backup since Ryu takes time to discover hosts
        if "hosts" in data:
            self.topology_metadata["static_hosts"] = data["hosts"]
        self.logger.info(f"Topology type updated: {self.topology_metadata['type']}")

    def get_topology(self):
        """
        Return Hybrid Topology:
        - Type: from static JSON
        - Switches/Links: from Live Ryu Discovery (LLDP)
        """
        
        # 1. Discover Switches Live
        # get_switch(self) returns list of topology.switches.Switch objects
        ryu_switches = get_switch(self, None)
        live_switches = [f"s{s.dp.id}" for s in ryu_switches]
        
        # 2. Discover Links Live
        # get_link(self) returns list of topology.switches.Link objects
        ryu_links = get_link(self, None)
        live_links = []
        for l in ryu_links:
            src = f"s{l.src.dpid}"
            dst = f"s{l.dst.dpid}"
            live_links.append([src, dst])
            
        # 3. Hosts (Combination)
        # Ryu discovers hosts only after they send packets
        # For better UX, we use the static list if available, or empty
        live_hosts = self.topology_metadata.get("static_hosts", [])
        
        # Note: We need to manually add links from Hosts to Switches if we rely on static host list
        # Because Ryu get_link() only returns Switch-to-Switch links.
        # To simplify: we will reuse static links that involve hosts ('h1'), 
        # but replace switch-switch links with live ones.
        
        final_links = []
        
        # Add live switch-switch links
        final_links.extend(live_links)
        
        # Add cached host links from the static metadata (since we can't discover them easily instantly)
        if "links" in self.topology_metadata:
            for link in self.topology_metadata["links"]:
                u, v = link[0], link[1]
                if u.startswith('h') or v.startswith('h'):
                    final_links.append(link)

        return {
            "type": self.topology_metadata["type"],
            "switches": live_switches,
            "hosts": live_hosts,
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