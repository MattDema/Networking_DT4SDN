# physical_controller.py
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
import json


class PhysicalTwinController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(PhysicalTwinController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        
        # Store topology metadata (received from Mininet script)
        self.topology_metadata = {"type": "Unknown", "switches": [], "links": []}
        
        # Live tracking of link status (dpid -> port_no -> is_live)
        self.port_status = {} 

        # Initialize WSGI for REST API
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

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def port_status_handler(self, ev):
        """Handle link up/down events"""
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        port = msg.desc
        
        dpid = dp.id
        port_no = port.port_no
        
        # Check if port is down (LINK_DOWN bit set)
        is_down = (port.state & ofp.OFPPS_LINK_DOWN)
        status = "DOWN" if is_down else "UP"
        
        self.logger.info(f"EVENT: Switch s{dpid} Port {port_no} is {status}")
        
        if dpid not in self.port_status:
            self.port_status[dpid] = {}
        
        self.port_status[dpid][port_no] = (not is_down)

    
    # --- Helper methods for REST Controller ---
    def set_topology(self, data):
        self.topology_metadata = data
        self.logger.info(f"Topology metadata updated via API: {data.get('type')}")

    def get_topology(self):
        """Return topology mixed with live status"""
        live_topo = self.topology_metadata.copy()
        
        # Create a new list for links to inject status
        updated_links = []
        
        # Original format from Mininet: [["s1", "s2", {}], ["s2", "s3", {}]] or similar
        # We need to map node names to dpid to check status
        # Note: This is an approximation. Mininet script sends names "s1", "s2".
        # We assume sX corresponds to dpid X.
        
        if 'links' in live_topo:
            for link in live_topo['links']:
                # link is usually [node1, node2, opts]
                node1, node2 = link[0], link[1]
                
                status = "UP"
                
                # Check if this link is affected by any DOWN port
                # We simply check if either side of the link has a down port? 
                # Ideally, we need to know WHICH port connects s1 to s2.
                # Since we don't have the full graph here, we will just mark the link
                # if we detect a port down event recently?
                
                # BETTER APPROACH FOR VISUALIZATION:
                # We return the raw 'port_status' to the Dashboard.
                # Let the Dashboard figure out which link is down visually.
                updated_links.append(link)
        
        live_topo['port_status'] = self.port_status
        return live_topo


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