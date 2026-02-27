#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from datetime import datetime


MODBUS_FIELDS = [
    "timestamp",
    "frame_time_epoch_raw",
    "frame_len",
    "frame_protocols",
    "has_vlan",
    "vlan_id",
    "src_ip",
    "dst_ip",
    "ip_ttl",
    "tcp_srcport",
    "tcp_dstport",
    "tcp_len",
    "tcp_window",
    "tcp_flags_ack",
    "tcp_flags_push",
    "tcp_flags_reset",
    "tcp_flags_syn",
    "frame_time_delta",
    "tcp_time_delta",
    "delta_source",
    "mbtcp_trans_id",
    "mbtcp_unit_id",
    "mbtcp_len",
    "modbus_func_code",
    "modbus_reference_num",
    "modbus_word_cnt",
    "modbus_exception",
    "modbus_response_time",
    "modbus_regval_uint16",
]


MQTT_FIELDS = [
    "timestamp",
    "frame_time_epoch_raw",
    "frame_len",
    "frame_protocols",
    "has_vlan",
    "vlan_id",
    "src_ip",
    "dst_ip",
    "ip_ttl",
    "tcp_srcport",
    "tcp_dstport",
    "tcp_len",
    "tcp_window",
    "tcp_flags_ack",
    "tcp_flags_push",
    "tcp_flags_reset",
    "tcp_flags_syn",
    "frame_time_delta",
    "tcp_time_delta",
    "delta_source",
    "mqtt_len",
    "mqtt_msgid",
    "mqtt_topic",
    "mqtt_topic_len",
    "mqtt_msgtype",
    "mqtt_qos",
    "mqtt_dupflag",
    "mqtt_retain",
    "mqtt_payload_hex",
    "mqtt_payload_len_bytes",
]

OPCUA_FIELDS = [
    "timestamp",
    "frame_time_epoch_raw",
    "frame_len",
    "frame_protocols",
    "has_vlan",
    "vlan_id",
    "src_ip",
    "dst_ip",
    "ip_ttl",
    "tcp_srcport",
    "tcp_dstport",
    "tcp_len",
    "tcp_window",
    "tcp_flags_ack",
    "tcp_flags_push",
    "tcp_flags_reset",
    "tcp_flags_syn",
    "frame_time_delta",
    "tcp_time_delta",
    "delta_source",
    "opcua_transport_type",
    "opcua_transport_chunk",
    "opcua_transport_size",
    "opcua_transport_scid",
    "opcua_security_tokenid",
]


S7COMM_FIELDS = [
    "timestamp",
    "frame_time_epoch_raw",
    "frame_len",
    "frame_protocols",
    "has_vlan",
    "vlan_id",
    "src_ip",
    "dst_ip",
    "ip_ttl",
    "tcp_srcport",
    "tcp_dstport",
    "tcp_len",
    "tcp_window",
    "tcp_flags_ack",
    "tcp_flags_push",
    "tcp_flags_reset",
    "tcp_flags_syn",
    "frame_time_delta",
    "tcp_time_delta",
    "delta_source",
    "tpkt_version",
    "tpkt_length",
    "cotp_type",
    "cotp_destref",
    "cotp_tpdu_number",
    "cotp_eot",
    "s7_header_protid",
    "s7_header_rosctr",
    "s7_header_redid",
    "s7_header_pduref",
    "s7_header_parlg",
    "s7_header_datlg",
    "s7_header_errcls",
    "s7_header_errcod",
    "s7_param_func",
    "s7_param_itemcount",
    "s7_param_pdu_length",
    "s7_param_maxamq_calling",
    "s7_param_maxamq_called",
    "s7_item_varspec",
    "s7_item_syntaxid",
    "s7_item_transp_size",
    "s7_item_length",
    "s7_item_db",
    "s7_item_area",
    "s7_item_address",
    "s7_data_returncode",
    "s7_data_transportsize",
    "s7_data_length",
    "s7_resp_data_hex",
    "s7_param_item_raw",
    "s7_data_item_raw",
]


PROFINET_FIELDS = [
    "timestamp",
    "frame_time_epoch_raw",
    "frame_len",
    "frame_protocols",
    "has_vlan",
    "vlan_id",
    "eth_src",
    "eth_dst",
    "eth_type",
    "src_ip",
    "dst_ip",
    "ip_ttl",
    "tcp_srcport",
    "tcp_dstport",
    "tcp_len",
    "tcp_window",
    "tcp_flags_ack",
    "tcp_flags_push",
    "tcp_flags_reset",
    "tcp_flags_syn",
    "frame_time_delta",
    "tcp_time_delta",
    "delta_source",
    "pn_rt_frame_id",
    "pn_ptcp_sequence_id",
    "pn_ptcp_delay1ns",
    "pn_ptcp_padding",
    "pn_ptcp_tl_type",
    "pn_ptcp_tl_length",
    "pn_ptcp_header_raw",
    "pn_ptcp_block_raw",
]


def _first(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val


def as_dict(val):
    val = _first(val)
    return val if isinstance(val, dict) else {}


def as_int(val, default=-1):
    val = _first(val)
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default


def as_float(val, default=0.0):
    val = _first(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def as_str(val, default=""):
    val = _first(val)
    if val is None:
        return default
    return str(val)


def as_json(val, default=""):
    val = _first(val)
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return default
    return str(val)


def to_epoch_seconds(val, default=0.0):
    val = _first(val)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        pass
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return default
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            return default
    return default


def extract_modbus_regval(modbus):
    for val in modbus.values():
        if isinstance(val, dict) and "modbus.regval_uint16" in val:
            return as_int(val.get("modbus.regval_uint16"), -1)
    return -1


def parse_modbus_row(obj):
    layers = as_dict(as_dict(obj.get("_source")).get("layers"))
    frame = as_dict(layers.get("frame"))
    ip = as_dict(layers.get("ip"))
    tcp = as_dict(layers.get("tcp"))
    tcp_flags = as_dict(tcp.get("tcp.flags_tree"))
    tcp_ts = as_dict(tcp.get("Timestamps"))
    mbtcp = as_dict(layers.get("mbtcp"))
    modbus = as_dict(layers.get("modbus"))
    vlan = as_dict(layers.get("vlan"))

    raw_frame_delta = frame.get("frame.time_delta")
    raw_tcp_delta = tcp_ts.get("tcp.time_delta")
    delta_source = "frame" if raw_frame_delta is not None else ("tcp" if raw_tcp_delta is not None else "none")

    frame_time_epoch_raw = _first(frame.get("frame.time_epoch"))
    return {
        "timestamp": to_epoch_seconds(frame_time_epoch_raw, 0.0),
        "frame_time_epoch_raw": as_str(frame_time_epoch_raw, ""),
        "frame_len": as_int(frame.get("frame.len"), -1),
        "frame_protocols": as_str(frame.get("frame.protocols"), ""),
        "has_vlan": 1 if "vlan" in layers else 0,
        "vlan_id": as_int(vlan.get("vlan.id"), -1),
        "src_ip": as_str(ip.get("ip.src"), ""),
        "dst_ip": as_str(ip.get("ip.dst"), ""),
        "ip_ttl": as_int(ip.get("ip.ttl"), -1),
        "tcp_srcport": as_int(tcp.get("tcp.srcport"), -1),
        "tcp_dstport": as_int(tcp.get("tcp.dstport"), -1),
        "tcp_len": as_int(tcp.get("tcp.len"), -1),
        "tcp_window": as_int(tcp.get("tcp.window_size"), -1),
        "tcp_flags_ack": as_int(tcp_flags.get("tcp.flags.ack"), -1),
        "tcp_flags_push": as_int(tcp_flags.get("tcp.flags.push"), -1),
        "tcp_flags_reset": as_int(tcp_flags.get("tcp.flags.reset"), -1),
        "tcp_flags_syn": as_int(tcp_flags.get("tcp.flags.syn"), -1),
        "frame_time_delta": as_float(raw_frame_delta, -1.0),
        "tcp_time_delta": as_float(raw_tcp_delta, -1.0),
        "delta_source": delta_source,
        "mbtcp_trans_id": as_int(mbtcp.get("mbtcp.trans_id"), -1),
        "mbtcp_unit_id": as_int(mbtcp.get("mbtcp.unit_id"), -1),
        "mbtcp_len": as_int(mbtcp.get("mbtcp.len"), -1),
        "modbus_func_code": as_int(modbus.get("modbus.func_code"), -1),
        "modbus_reference_num": as_int(modbus.get("modbus.reference_num"), -1),
        "modbus_word_cnt": as_int(modbus.get("modbus.word_cnt"), -1),
        "modbus_exception": as_int(modbus.get("modbus.exception"), -1),
        "modbus_response_time": as_float(modbus.get("modbus.response_time"), -1.0),
        "modbus_regval_uint16": extract_modbus_regval(modbus),
    }


def parse_mqtt_row(obj):
    layers = as_dict(as_dict(obj.get("_source")).get("layers"))
    frame = as_dict(layers.get("frame"))
    ip = as_dict(layers.get("ip"))
    tcp = as_dict(layers.get("tcp"))
    tcp_flags = as_dict(tcp.get("tcp.flags_tree"))
    tcp_ts = as_dict(tcp.get("Timestamps"))
    mqtt = as_dict(layers.get("mqtt"))
    mqtt_flags = as_dict(mqtt.get("mqtt.hdrflags_tree"))
    vlan = as_dict(layers.get("vlan"))

    raw_frame_delta = frame.get("frame.time_delta")
    raw_tcp_delta = tcp_ts.get("tcp.time_delta")
    delta_source = "frame" if raw_frame_delta is not None else ("tcp" if raw_tcp_delta is not None else "none")

    payload_hex = as_str(mqtt.get("mqtt.msg"), "")
    payload_len = len(payload_hex.split(":")) if payload_hex else 0

    frame_time_epoch_raw = _first(frame.get("frame.time_epoch"))
    return {
        "timestamp": to_epoch_seconds(frame_time_epoch_raw, 0.0),
        "frame_time_epoch_raw": as_str(frame_time_epoch_raw, ""),
        "frame_len": as_int(frame.get("frame.len"), -1),
        "frame_protocols": as_str(frame.get("frame.protocols"), ""),
        "has_vlan": 1 if "vlan" in layers else 0,
        "vlan_id": as_int(vlan.get("vlan.id"), -1),
        "src_ip": as_str(ip.get("ip.src"), ""),
        "dst_ip": as_str(ip.get("ip.dst"), ""),
        "ip_ttl": as_int(ip.get("ip.ttl"), -1),
        "tcp_srcport": as_int(tcp.get("tcp.srcport"), -1),
        "tcp_dstport": as_int(tcp.get("tcp.dstport"), -1),
        "tcp_len": as_int(tcp.get("tcp.len"), -1),
        "tcp_window": as_int(tcp.get("tcp.window_size"), -1),
        "tcp_flags_ack": as_int(tcp_flags.get("tcp.flags.ack"), -1),
        "tcp_flags_push": as_int(tcp_flags.get("tcp.flags.push"), -1),
        "tcp_flags_reset": as_int(tcp_flags.get("tcp.flags.reset"), -1),
        "tcp_flags_syn": as_int(tcp_flags.get("tcp.flags.syn"), -1),
        "frame_time_delta": as_float(raw_frame_delta, -1.0),
        "tcp_time_delta": as_float(raw_tcp_delta, -1.0),
        "delta_source": delta_source,
        "mqtt_len": as_int(mqtt.get("mqtt.len"), -1),
        "mqtt_msgid": as_int(mqtt.get("mqtt.msgid"), -1),
        "mqtt_topic": as_str(mqtt.get("mqtt.topic"), ""),
        "mqtt_topic_len": as_int(mqtt.get("mqtt.topic_len"), -1),
        "mqtt_msgtype": as_int(mqtt_flags.get("mqtt.msgtype"), -1),
        "mqtt_qos": as_int(mqtt_flags.get("mqtt.qos"), -1),
        "mqtt_dupflag": as_int(mqtt_flags.get("mqtt.dupflag"), -1),
        "mqtt_retain": as_int(mqtt_flags.get("mqtt.retain"), -1),
        "mqtt_payload_hex": payload_hex,
        "mqtt_payload_len_bytes": payload_len,
    }


def parse_opcua_row(obj):
    layers = as_dict(as_dict(obj.get("_source")).get("layers"))
    frame = as_dict(layers.get("frame"))
    ip = as_dict(layers.get("ip"))
    tcp = as_dict(layers.get("tcp"))
    tcp_flags = as_dict(tcp.get("tcp.flags_tree"))
    tcp_ts = as_dict(tcp.get("Timestamps"))
    opcua = as_dict(layers.get("opcua"))
    vlan = as_dict(layers.get("vlan"))

    raw_frame_delta = frame.get("frame.time_delta")
    raw_tcp_delta = tcp_ts.get("tcp.time_delta")
    delta_source = "frame" if raw_frame_delta is not None else ("tcp" if raw_tcp_delta is not None else "none")

    frame_time_epoch_raw = _first(frame.get("frame.time_epoch"))
    return {
        "timestamp": to_epoch_seconds(frame_time_epoch_raw, 0.0),
        "frame_time_epoch_raw": as_str(frame_time_epoch_raw, ""),
        "frame_len": as_int(frame.get("frame.len"), -1),
        "frame_protocols": as_str(frame.get("frame.protocols"), ""),
        "has_vlan": 1 if "vlan" in layers else 0,
        "vlan_id": as_int(vlan.get("vlan.id"), -1),
        "src_ip": as_str(ip.get("ip.src"), ""),
        "dst_ip": as_str(ip.get("ip.dst"), ""),
        "ip_ttl": as_int(ip.get("ip.ttl"), -1),
        "tcp_srcport": as_int(tcp.get("tcp.srcport"), -1),
        "tcp_dstport": as_int(tcp.get("tcp.dstport"), -1),
        "tcp_len": as_int(tcp.get("tcp.len"), -1),
        "tcp_window": as_int(tcp.get("tcp.window_size"), -1),
        "tcp_flags_ack": as_int(tcp_flags.get("tcp.flags.ack"), -1),
        "tcp_flags_push": as_int(tcp_flags.get("tcp.flags.push"), -1),
        "tcp_flags_reset": as_int(tcp_flags.get("tcp.flags.reset"), -1),
        "tcp_flags_syn": as_int(tcp_flags.get("tcp.flags.syn"), -1),
        "frame_time_delta": as_float(raw_frame_delta, -1.0),
        "tcp_time_delta": as_float(raw_tcp_delta, -1.0),
        "delta_source": delta_source,
        "opcua_transport_type": as_str(opcua.get("opcua.transport.type"), ""),
        "opcua_transport_chunk": as_str(opcua.get("opcua.transport.chunk"), ""),
        "opcua_transport_size": as_int(opcua.get("opcua.transport.size"), -1),
        "opcua_transport_scid": as_int(opcua.get("opcua.transport.scid"), -1),
        "opcua_security_tokenid": as_int(opcua.get("opcua.security.tokenid"), -1),
    }


def parse_s7comm_row(obj):
    layers = as_dict(as_dict(obj.get("_source")).get("layers"))
    frame = as_dict(layers.get("frame"))
    ip = as_dict(layers.get("ip"))
    tcp = as_dict(layers.get("tcp"))
    tcp_flags = as_dict(tcp.get("tcp.flags_tree"))
    tcp_ts = as_dict(tcp.get("Timestamps"))
    tpkt = as_dict(layers.get("tpkt"))
    cotp = as_dict(layers.get("cotp"))
    s7comm = as_dict(layers.get("s7comm"))
    s7_header = as_dict(s7comm.get("s7comm.header"))
    s7_param = as_dict(s7comm.get("s7comm.param"))
    s7_data = as_dict(s7comm.get("s7comm.data"))
    s7_param_item_raw = _first(s7_param.get("s7comm.param.item"))
    s7_data_item_raw = _first(s7_data.get("s7comm.data.item"))
    s7_param_item = s7_param_item_raw if isinstance(s7_param_item_raw, dict) else {}
    s7_data_item = s7_data_item_raw if isinstance(s7_data_item_raw, dict) else {}
    vlan = as_dict(layers.get("vlan"))

    raw_frame_delta = frame.get("frame.time_delta")
    raw_tcp_delta = tcp_ts.get("tcp.time_delta")
    delta_source = "frame" if raw_frame_delta is not None else ("tcp" if raw_tcp_delta is not None else "none")

    frame_time_epoch_raw = _first(frame.get("frame.time_epoch"))
    return {
        "timestamp": to_epoch_seconds(frame_time_epoch_raw, 0.0),
        "frame_time_epoch_raw": as_str(frame_time_epoch_raw, ""),
        "frame_len": as_int(frame.get("frame.len"), -1),
        "frame_protocols": as_str(frame.get("frame.protocols"), ""),
        "has_vlan": 1 if "vlan" in layers else 0,
        "vlan_id": as_int(vlan.get("vlan.id"), -1),
        "src_ip": as_str(ip.get("ip.src"), ""),
        "dst_ip": as_str(ip.get("ip.dst"), ""),
        "ip_ttl": as_int(ip.get("ip.ttl"), -1),
        "tcp_srcport": as_int(tcp.get("tcp.srcport"), -1),
        "tcp_dstport": as_int(tcp.get("tcp.dstport"), -1),
        "tcp_len": as_int(tcp.get("tcp.len"), -1),
        "tcp_window": as_int(tcp.get("tcp.window_size"), -1),
        "tcp_flags_ack": as_int(tcp_flags.get("tcp.flags.ack"), -1),
        "tcp_flags_push": as_int(tcp_flags.get("tcp.flags.push"), -1),
        "tcp_flags_reset": as_int(tcp_flags.get("tcp.flags.reset"), -1),
        "tcp_flags_syn": as_int(tcp_flags.get("tcp.flags.syn"), -1),
        "frame_time_delta": as_float(raw_frame_delta, -1.0),
        "tcp_time_delta": as_float(raw_tcp_delta, -1.0),
        "delta_source": delta_source,
        "tpkt_version": as_str(tpkt.get("tpkt.version"), ""),
        "tpkt_length": as_str(tpkt.get("tpkt.length"), ""),
        "cotp_type": as_str(cotp.get("cotp.type"), ""),
        "cotp_destref": as_str(cotp.get("cotp.destref"), ""),
        "cotp_tpdu_number": as_str(cotp.get("cotp.tpdu-number"), ""),
        "cotp_eot": as_str(cotp.get("cotp.eot"), ""),
        "s7_header_protid": as_str(s7_header.get("s7comm.header.protid"), ""),
        "s7_header_rosctr": as_str(s7_header.get("s7comm.header.rosctr"), ""),
        "s7_header_redid": as_str(s7_header.get("s7comm.header.redid"), ""),
        "s7_header_pduref": as_str(s7_header.get("s7comm.header.pduref"), ""),
        "s7_header_parlg": as_str(s7_header.get("s7comm.header.parlg"), ""),
        "s7_header_datlg": as_str(s7_header.get("s7comm.header.datlg"), ""),
        "s7_header_errcls": as_str(s7_header.get("s7comm.header.errcls"), ""),
        "s7_header_errcod": as_str(s7_header.get("s7comm.header.errcod"), ""),
        "s7_param_func": as_str(s7_param.get("s7comm.param.func"), ""),
        "s7_param_itemcount": as_str(s7_param.get("s7comm.param.itemcount"), ""),
        "s7_param_pdu_length": as_str(s7_param.get("s7comm.param.pdu_length"), ""),
        "s7_param_maxamq_calling": as_str(s7_param.get("s7comm.param.maxamq_calling"), ""),
        "s7_param_maxamq_called": as_str(s7_param.get("s7comm.param.maxamq_called"), ""),
        "s7_item_varspec": as_str(s7_param_item.get("s7comm.param.item.varspec"), ""),
        "s7_item_syntaxid": as_str(s7_param_item.get("s7comm.param.item.syntaxid"), ""),
        "s7_item_transp_size": as_str(s7_param_item.get("s7comm.param.item.transp_size"), ""),
        "s7_item_length": as_str(s7_param_item.get("s7comm.param.item.length"), ""),
        "s7_item_db": as_str(s7_param_item.get("s7comm.param.item.db"), ""),
        "s7_item_area": as_str(s7_param_item.get("s7comm.param.item.area"), ""),
        "s7_item_address": as_str(s7_param_item.get("s7comm.param.item.address"), ""),
        "s7_data_returncode": as_str(s7_data_item.get("s7comm.data.returncode"), ""),
        "s7_data_transportsize": as_str(s7_data_item.get("s7comm.data.transportsize"), ""),
        "s7_data_length": as_str(s7_data_item.get("s7comm.data.length"), ""),
        "s7_resp_data_hex": as_str(s7_data_item.get("s7comm.resp.data"), ""),
        "s7_param_item_raw": as_json(s7_param_item_raw, ""),
        "s7_data_item_raw": as_json(s7_data_item_raw, ""),
    }


def parse_profinet_row(obj):
    layers = as_dict(as_dict(obj.get("_source")).get("layers"))
    frame = as_dict(layers.get("frame"))
    eth = as_dict(layers.get("eth"))
    ip = as_dict(layers.get("ip"))
    tcp = as_dict(layers.get("tcp"))
    tcp_flags = as_dict(tcp.get("tcp.flags_tree"))
    tcp_ts = as_dict(tcp.get("Timestamps"))
    vlan = as_dict(layers.get("vlan"))
    pn_rt = as_dict(layers.get("pn_rt"))
    pn_ptcp = as_dict(layers.get("pn_ptcp"))
    pn_ptcp_header_raw = _first(pn_ptcp.get("pn_ptcp.header"))
    pn_ptcp_block_raw = _first(pn_ptcp.get("pn_ptcp.block"))
    pn_ptcp_header = pn_ptcp_header_raw if isinstance(pn_ptcp_header_raw, dict) else {}
    pn_ptcp_block = pn_ptcp_block_raw if isinstance(pn_ptcp_block_raw, dict) else {}
    pn_ptcp_tlv = as_dict(pn_ptcp_block.get("pn_ptcp.tlvheader"))

    raw_frame_delta = frame.get("frame.time_delta")
    raw_tcp_delta = tcp_ts.get("tcp.time_delta")
    delta_source = "frame" if raw_frame_delta is not None else ("tcp" if raw_tcp_delta is not None else "none")

    frame_time_epoch_raw = _first(frame.get("frame.time_epoch"))
    return {
        "timestamp": to_epoch_seconds(frame_time_epoch_raw, 0.0),
        "frame_time_epoch_raw": as_str(frame_time_epoch_raw, ""),
        "frame_len": as_int(frame.get("frame.len"), -1),
        "frame_protocols": as_str(frame.get("frame.protocols"), ""),
        "has_vlan": 1 if "vlan" in layers else 0,
        "vlan_id": as_int(vlan.get("vlan.id"), -1),
        "eth_src": as_str(eth.get("eth.src"), ""),
        "eth_dst": as_str(eth.get("eth.dst"), ""),
        "eth_type": as_str(eth.get("eth.type"), ""),
        "src_ip": as_str(ip.get("ip.src"), ""),
        "dst_ip": as_str(ip.get("ip.dst"), ""),
        "ip_ttl": as_int(ip.get("ip.ttl"), -1),
        "tcp_srcport": as_int(tcp.get("tcp.srcport"), -1),
        "tcp_dstport": as_int(tcp.get("tcp.dstport"), -1),
        "tcp_len": as_int(tcp.get("tcp.len"), -1),
        "tcp_window": as_int(tcp.get("tcp.window_size"), -1),
        "tcp_flags_ack": as_int(tcp_flags.get("tcp.flags.ack"), -1),
        "tcp_flags_push": as_int(tcp_flags.get("tcp.flags.push"), -1),
        "tcp_flags_reset": as_int(tcp_flags.get("tcp.flags.reset"), -1),
        "tcp_flags_syn": as_int(tcp_flags.get("tcp.flags.syn"), -1),
        "frame_time_delta": as_float(raw_frame_delta, -1.0),
        "tcp_time_delta": as_float(raw_tcp_delta, -1.0),
        "delta_source": delta_source,
        "pn_rt_frame_id": as_str(pn_rt.get("pn_rt.frame_id"), ""),
        "pn_ptcp_sequence_id": as_str(pn_ptcp_header.get("pn_ptcp.sequence_id"), ""),
        "pn_ptcp_delay1ns": as_str(pn_ptcp.get("pn_ptcp.delay1ns"), ""),
        "pn_ptcp_padding": as_str(pn_ptcp_header.get("pn.padding"), ""),
        "pn_ptcp_tl_type": as_str(pn_ptcp_tlv.get("pn_ptcp.tl_type"), ""),
        "pn_ptcp_tl_length": as_str(pn_ptcp_tlv.get("pn_ptcp.tl_length"), ""),
        "pn_ptcp_header_raw": as_json(pn_ptcp_header_raw, ""),
        "pn_ptcp_block_raw": as_json(pn_ptcp_block_raw, ""),
    }


def detect_input_mode(path):
    with open(path, "rb") as f:
        head = f.read(64 * 1024)
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:]
    if b"\x00" in head:
        head = head.replace(b"\x00", b"")
    for b in head:
        if chr(b).isspace():
            continue
        return "json_array" if b == ord("[") else "jsonl"
    return "jsonl"


def iter_json_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in ("[", "]", ","):
                continue
            if line.endswith(","):
                line = line[:-1].rstrip()
            if not line or line in ("[", "]"):
                continue
            yield json.loads(line)


def iter_json_array(path, chunk_size_mb=8):
    decoder = json.JSONDecoder()
    chunk_size = chunk_size_mb * 1024 * 1024
    buffer = ""
    eof = False
    in_array = False
    need_more = True

    with open(path, "rb", buffering=chunk_size) as f:
        while True:
            if not eof and (need_more or len(buffer) < chunk_size):
                chunk = f.read(chunk_size)
                if chunk:
                    if b"\x00" in chunk:
                        chunk = chunk.replace(b"\x00", b"")
                    buffer += chunk.decode("utf-8", errors="ignore")
                    need_more = False
                else:
                    eof = True

            if not in_array:
                idx = buffer.find("[")
                if idx == -1:
                    if eof:
                        return
                    buffer = ""
                    continue
                buffer = buffer[idx + 1 :]
                in_array = True

            i = 0
            blen = len(buffer)
            while i < blen and buffer[i] in " \r\n\t,":
                i += 1
            if i:
                buffer = buffer[i:]

            if not buffer:
                if eof:
                    return
                need_more = True
                continue

            if buffer[0] == "]":
                return

            try:
                obj, idx = decoder.raw_decode(buffer)
                yield obj
                buffer = buffer[idx:]
                need_more = False
            except json.JSONDecodeError:
                if eof:
                    return
                need_more = True


def iter_input_objects(path):
    mode = detect_input_mode(path)
    if mode == "json_array":
        yield from iter_json_array(path)
    else:
        yield from iter_json_lines(path)


def convert_packets(inp, outp, protocol, progress_every=200000):
    if protocol == "modbus":
        parser = parse_modbus_row
        fields = MODBUS_FIELDS
    elif protocol == "mqtt":
        parser = parse_mqtt_row
        fields = MQTT_FIELDS
    elif protocol == "opcua":
        parser = parse_opcua_row
        fields = OPCUA_FIELDS
    elif protocol == "s7comm":
        parser = parse_s7comm_row
        fields = S7COMM_FIELDS
    elif protocol == "profinet":
        parser = parse_profinet_row
        fields = PROFINET_FIELDS
    else:
        raise ValueError(f"unsupported protocol: {protocol}")

    total = 0
    bad = 0
    with open(outp, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for obj in iter_input_objects(inp):
            total += 1
            try:
                row = parser(obj)
                writer.writerow(row)
            except Exception:
                bad += 1
            if progress_every > 0 and total % progress_every == 0:
                print(f"[{protocol}] processed={total:,} bad={bad:,}", file=sys.stderr, flush=True)
    return total, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["modbus", "mqtt", "opcua", "s7comm", "profinet"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--progress-every", type=int, default=200000)
    args = ap.parse_args()

    total, bad = convert_packets(args.inp, args.outp, args.protocol, args.progress_every)
    print(json.dumps({"protocol": args.protocol, "input": args.inp, "output": args.outp, "total": total, "bad": bad}, ensure_ascii=False))


if __name__ == "__main__":
    main()
