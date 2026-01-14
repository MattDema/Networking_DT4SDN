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

# --- STP IMPORTS ---
from ryu.lib import stplib
from ryu.lib import stp

class PhysicalTwinController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    # Diciamo a Ryu che vogliamo usare il contesto WSGI e la libreria STP
    _CONTEXTS = {
        'wsgi': WSGIApplication,
        'stplib': stplib.Stp
    }

    def __init__(self, *args, **kwargs):
        super(PhysicalTwinController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.stp = kwargs['stplib']  # Recuperiamo l'istanza STP
        
        # We still keep the metadata for the "Type" (e.g. "Linear")
        # But switches and links will be overwritten by live discovery
        self.topology_metadata = {"type": "Unknown", "switches": [], "links": [], "hosts": []}

        wsgi = kwargs['wsgi']
        wsgi.register(TopologyController, {'physical_controller': self})

        # Configurazione STP di base (opzionale: assegna priorità di default)
        # Questo aiuta a stabilizzare l'elezione del root bridge
        # config = {dpid: {'bridge_priority': 32768} for dpid in range(1, 20)}
        # self.stp.set_config(config)

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

    def delete_flow(self, datapath):
        """Cancella tutti i flussi di uno switch (utile quando STP cambia topologia)"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        mod = parser.OFPFlowMod(datapath, command=ofproto.OFPFC_DELETE,
                                out_port=ofproto.OFPP_ANY,
                                out_group=ofproto.OFPG_ANY,
                                match=match)
        datapath.send_msg(mod)

    # --- GESTORE CAMBIO STATO PORTA (STP) ---
    @set_ev_cls(stplib.EventTopologyChange, MAIN_DISPATCHER)
    def _topology_change_handler(self, ev):
        dp = ev.dp
        dpid_str = dpid_lib.dpid_to_str(dp.id)
        msg = 'Port state changed in switch: %s' % dpid_str
        self.logger.debug(msg)

        if dp.id in self.mac_to_port:
            del self.mac_to_port[dp.id]
        self.delete_flow(dp)

    # --- GESTORE PACKET IN (MODIFICATO PER STP) ---
    # Usiamo stplib.EventPacketIn invece di ofp_event.EventOFPPacketIn
    # Questo assicura che riceveremo pacchetti SOLO se la porta è in stato FORWARD
    @set_ev_cls(stplib.EventPacketIn, MAIN_DISPATCHER)
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
        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            # verify if we have a valid buffer_id, if yes avoid to send both
            # flow_mod & packet_out
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

    # --- Helper methods for REST Controller (INVARIATI) ---
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
        
        # 2. Discover Links Live (S-S only) - DEDUPLICATED
        ryu_links = get_link(self, None)
        live_links = []
        seen_pairs = set()

        for l in ryu_links:
            src = f"s{l.src.dpid}"
            dst = f"s{l.dst.dpid}"
            
            # Create a sorted tuple representing the link regardless of direction
            link_pair = tuple(sorted((src, dst)))
            
            if link_pair not in seen_pairs:
                live_links.append([src, dst])
                seen_pairs.add(link_pair)
            
        # 3. Hosts & Host Links (Static Fallback)
        final_hosts = self.topology_metadata.get("hosts", [])
        static_links = self.topology_metadata.get("links", [])
        static_switches_list = self.topology_metadata.get("switches", [])
        
        # Start with live links
        final_links = list(live_links)
        
        # Merge Host Links from Static Metadata
        for link in static_links:
            u, v = link[0], link[1]
            u_is_switch = (u in static_switches_list)
            v_is_switch = (v in static_switches_list)
            if not (u_is_switch and v_is_switch):
                final_links.append(link)

        return {
            "type": self.topology_metadata.get("type", "Unknown"),
            "switches": live_switches if live_switches else static_switches_list,
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
        return Response(content_type='application/json', charset='utf-8', body=body)

    @route('topology', '/topology/metadata', methods=['POST'])
    def set_metadata(self, req, **kwargs):
        """Mininet script calls this to set current topology info"""
        try:
            data = json.loads(req.body) if req.body else {}
            self.physical_controller.set_topology(data)
            return Response(content_type='application/json', charset='utf-8', body=json.dumps({"status": "success"}))
        except Exception as e:
            return Response(status=500, body=str(e))