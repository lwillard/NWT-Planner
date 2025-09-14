# -*- coding: utf-8 -*-
def classFactory(iface):
    from .nwt_planner_plugin import NuclearWarTargetingPlannerPlugin
    return NuclearWarTargetingPlannerPlugin(iface)
