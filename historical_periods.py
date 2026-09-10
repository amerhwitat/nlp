"""Period taxonomy for the ancient-object catalog.

Dates are broad display metadata; use site-specific chronology when known.
"""
PERIODS = [
    {"key":"paleolithic","name":"Paleolithic","parent":"prehistory","start":"-2500000","end":"-40000","notes":"Lower/Middle/Upper Paleolithic umbrella."},
    {"key":"epipaleolithic","name":"Epipaleolithic","parent":"prehistory","start":"-40000","end":"-12000","notes":"Regional transition terminology; dates vary."},
    {"key":"neolithic","name":"Neolithic","parent":"prehistory","start":"-12000","end":"-6000","notes":"Regional chronology varies."},
    {"key":"chalcolithic","name":"Chalcolithic","parent":"ancient","start":"-6000","end":"-3500","notes":"Copper/stone transition; regional chronology varies."},
    {"key":"bronze_age","name":"Bronze Age","parent":"ancient","start":"-3500","end":"-1250","notes":"Broad Near Eastern chronology."},
    {"key":"iron_age","name":"Iron Age","parent":"ancient","start":"-1250","end":"-333","notes":"Broad regional chronology."},
    {"key":"hellenistic","name":"Hellenistic / Greek","parent":"classical","start":"-333","end":"-65","notes":"Use local political and archaeological context."},
    {"key":"roman","name":"Roman","parent":"classical","start":"-65","end":"324","notes":"Roman-period Arabia and neighboring regions."},
    {"key":"byzantine","name":"Byzantine / Eastern Roman","parent":"late_antique","start":"324","end":"632","notes":"Broad eastern Roman chronology."},
    {"key":"early_islamic","name":"Early Islamic","parent":"islamic","start":"632","end":"900","notes":"Use dynasty and site metadata for precision."},
    {"key":"medieval","name":"Medieval","parent":"islamic","start":"900","end":"1500","notes":"Broad cross-regional label."},
    {"key":"early_modern","name":"Early Modern","parent":"modern_era","start":"1500","end":"1800","notes":"Broad cross-regional label."},
    {"key":"modern","name":"Modern","parent":"modern_era","start":"1800","end":None,"notes":"Modern and contemporary material."},
]

def period_tree():
    nodes = {k:{"key":k,"name":k.replace("_"," ").title(),"children":[]} for k in ("prehistory","ancient","classical","late_antique","islamic","modern_era")}
    for p in PERIODS:
        nodes[p["parent"]]["children"].append(p)
    return nodes

def get_period(key):
    for period in PERIODS:
        if period["key"] == key:
            return period
    raise KeyError(key)
