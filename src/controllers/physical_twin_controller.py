from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
import json

from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link, get_host
from ryu.lib import dpid as dpid_lib

class PhysicalTwinController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    _CONTEXTS = {
        'wsgi': WSGIApplication
    }

    def __init__(self, *args, **kwargs):
        super(PhysicalTwinController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        
        self.topology_metadata = {"type": "Unknown", "switches": [], "links": [], "hosts": []}

        wsgi = kwargs['wsgi']
        wsgi.register(TopologyController, {'physical_controller': self})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install table-miss flow entry
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

    # standard packet-in handler no stp
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

        # learn a mac address to avoid flood next time
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
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

    # helper methods for rest controller
    def set_topology(self, data):
        self.topology_metadata = data
        self.logger.info(f"Topology metadata updated via API: {data.get('type')}")

    def get_topology(self):
        """
        Return Hybrid Topology:
        - Switches/Links (S-S): from Live Ryu Discovery (LLDP)
        - Hosts/Link (H-S): from static JSON (cached)
        """
        
        # 1. discover switches
        ryu_switches = get_switch(self, None)
        live_switches = [f"s{s.dp.id}" for s in ryu_switches]
        
        # 2. discover links (s-s only)
        ryu_links = get_link(self, None)
        live_links = []
        seen_pairs = set()

        for l in ryu_links:
            src = f"s{l.src.dpid}"
            dst = f"s{l.dst.dpid}"
            
            # create a sorted tuple representing the link regardless of direction
            link_pair = tuple(sorted((src, dst)))
            
            if link_pair not in seen_pairs:
                live_links.append([src, dst])
                seen_pairs.add(link_pair)
            
        # 3. hosts & host links
        final_hosts = self.topology_metadata.get("hosts", [])
        static_links = self.topology_metadata.get("links", [])
        static_switches_list = self.topology_metadata.get("switches", [])
        
        # start with live links
        final_links = list(live_links)
        
        # merge host links from static metadata
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
        """dashboard calls this to get current topology info"""
        body = json.dumps(self.physical_controller.get_topology())
        return Response(content_type='application/json', charset='utf-8', body=body)

    @route('topology', '/topology/metadata', methods=['POST'])
    def set_metadata(self, req, **kwargs):
        """mininet script calls this to set current topology info"""
        try:
            data = json.loads(req.body) if req.body else {}
            self.physical_controller.set_topology(data)
            return Response(content_type='application/json', charset='utf-8', body=json.dumps({"status": "success"}))
        except Exception as e:
            return Response(status=500, body=str(e))

