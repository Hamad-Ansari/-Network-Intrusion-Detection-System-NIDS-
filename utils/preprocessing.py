import pandas as pd
import numpy as np
import pickle
import streamlit as st

EXPECTED_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

CAT_COLS = ['protocol_type', 'service', 'flag']

PROTOCOL_TYPES = ['tcp', 'udp', 'icmp']
FLAG_TYPES = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'SH', 'S1']
SERVICE_TYPES = [
    'private', 'ecr_i', 'http', 'domain_u', 'other', 'smtp', 'finger', 'auth',
    'efs', 'ftp_data', 'daytime', 'courier', 'ssh', 'domain', 'shell', 'pop_2',
    'ntp_u', 'csnet_ns', 'bgp', 'telnet', 'tftp_u', 'ftp', 'X11', 'nntp',
    'discard', 'echo', 'eco_i', 'supdup', 'ctf', 'name', 'sunrpc', 'systat',
    'netstat', 'IRC', 'time', 'urp_i', 'login', 'printer', 'remote_job',
    'http_443', 'ldap', 'iso_tsap', 'pop_3', 'hostnames', 'vmnet', 'rje',
    'whois', 'imap4', 'gopher', 'uucp_path', 'netbios_ns', 'nnsp', 'mtp',
    'Z39_50', 'uucp', 'kshell', 'sql_net', 'link', 'exec', 'netbios_dgm',
    'netbios_ssn', 'klogin'
]


@st.cache_resource
def load_preprocessors(path='models/preprocessors.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def validate_columns(df):
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    return len(missing) == 0, missing


def preprocess_dataframe(df, preprocessors):
    df = df.copy()
    df = df[[c for c in EXPECTED_FEATURES if c in df.columns]]
    le_dict = preprocessors['le_dict']
    for col in CAT_COLS:
        if col in df.columns:
            le = le_dict[col]
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            df[col] = le.transform(df[col])
    df = df.fillna(0)
    scaler = preprocessors['scaler']
    X_scaled = scaler.transform(df)
    return X_scaled


def preprocess_manual_input(input_dict, preprocessors):
    df = pd.DataFrame([input_dict])
    return preprocess_dataframe(df, preprocessors)
