#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Copyright 2017-2019 Airinnova AB and the PyTornado authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------

# Authors:
# * Alessandro Gastaldi
# * Aaron Dettmann

"""
Functions for conversion of CPACS aircraft definition to native model

Developed at Airinnova AB, Stockholm, Sweden.
"""

import os
import logging
import numpy as np

from pytornado.objects.model import ComponentDefinitionError
from pytornado.objects.objecttools import all_controls, all_wings

logger = logging.getLogger(__name__)

try:
    import tixi.tixiwrapper as tixiwrapper
except ImportError:
    TIXI_INSTALLED = False
else:
    TIXI_INSTALLED = True

try:
    import tigl.tiglwrapper as tiglwrapper
except ImportError:
    TIGL_INSTALLED = False
else:
    TIGL_INSTALLED = True


NODE_NAME = '/cpacs/vehicles/aircraft/model'
NODE_REFS = '/cpacs/vehicles/aircraft/model/reference'
NODE_WINGS = '/cpacs/vehicles/aircraft/model/wings'
NODE_AIRFOILS = '/cpacs/vehicles/profiles/wingAirfoils'
NODE_AEROPMAP = '/cpacs/vehicles/aircraft/analyes/aeroPerformanceMap'
NODE_CONTROL = NODE_WINGS \
        + '/wing[{0:d}]/componentSegments/componentSegment[{1:d}]' \
        + '/controlSurfaces/{3:s}EdgeDevices/{3:s}EdgeDevice[{2:d}]'

NODE_TOOLSPEC = '/cpacs/toolspecific/pyTornado'
NODE_TS_CONTROL = NODE_TOOLSPEC + '/controlDevices'

COORD_FORMAT = '%+.7f'

# Airfoil bounding box
BOUNDING_BOX = '{:+.7} {:+.7} {:+.7} {:+.7}'.format(-1.0, -1.0, +2.5, +1.5)


def parse_str(entry, allow_char='-_', allow_none=False, allow_bool=False):
    """
    Convert input to valid STRING. Optionally, to TRUE, FALSE or NONE.

    Alphanum chars a...z, A...Z, 0...9 are always accepted.
    Additional valid chars can be passed :wT_CHAR.

        * ALLOW_NONE lets function return NONE if entry is 'NONE'.
        * ALLOW_BOOL lets function return TRUE/FALSE if entry is 'TRUE'/'FALSE'.

    Args:
        :entry: (string) input STRING
        :allow_char: (string) additional accepted characters (default: underscore & hyphen)
        :allow_none: (bool) detect when input is NONE (default: FALSE)
        :allow_bool: (bool) detect when input is TRUE or FALSE (default: FALSE)

    Returns:
        (?) converted STRING, or NONE, TRUE, FALSE
    """

    if not isinstance(entry, str):
        return ''
    elif allow_none and entry.upper().strip() == 'NONE':
        return None
    elif allow_bool and entry.upper().strip() == 'TRUE':
        return True
    elif allow_bool and entry.upper().strip() == 'FALSE':
        return False
    else:
        return ''.join([c for c in entry if c.isalnum() or c in allow_char])


def load(aircraft, state, settings):
    """
    Get aircraft model and flight state data from CPACS definition.

    Expects CPACS file in the AIRCRAFT folder of the WKDIR.
    Expects airfoil coordinate files in the AIRFOILS folder of the WKDIR.
    If missing, airfoil coordinates are extracted from the CPACS file.

    Args:
        :aircraft: (object) data structure for aircraft model
        :state: (object) data structure for flight state
        :settings: (object) data structure for execution settings
    """

    logger.debug("checking TIXI installation...")

    if not TIXI_INSTALLED:
        logger.error("Could not import 'tixiwrapper'.")
        raise ModuleNotFoundError("Module 'tixiwrapper' not found")

    logger.debug("TIXI imported.")
    logger.debug("Checking TIGL installation...")

    if not TIGL_INSTALLED:
        logger.error("Could not import 'tiglwrapper'")
        raise ModuleNotFoundError("Module 'tiglwrapper' not found")

    logger.debug("TIGL imported.")

    # Handles to TIXI, TIGL libraries
    tixi = tixiwrapper.Tixi()
    tigl = tiglwrapper.Tigl()

    filepath = settings.files['aircraft']
    logger.info(f"Loading aircraft from CPACS file: {filepath}...")

    if not os.path.exists(filepath):
        return logger.error(f"File '{filepath}' not found")

    tixi.open(filepath)

    # From Tigl documentation:
    # The UID of the configuration that should be loaded by TIGL. Could be NULL
    # or an empty string if the data set contains only one configuration.
    tigl.open(tixi, '')

    aircraft.reset()

    if tixi.checkElement(NODE_NAME):
        aircraft.uid = parse_str(tixi.getTextAttribute(NODE_NAME, 'uID'))
    else:
        logger.warning(f"Could not find path '{NODE_NAME}'")

    logger.debug("Loading aircraft '{aircraft.uid}'")
    logger.info("Loading aircraft wings...")

    if not tixi.checkElement(NODE_WINGS):
        return logger.error(f"Could not find path '{NODE_WINGS}'")

    # enumerate wings
    for idx_wing in range(1, tixi.getNamedChildrenCount(NODE_WINGS, 'wing') + 1):
        node_wing = NODE_WINGS + '/wing[{}]'.format(idx_wing)

        try:
            wing_uid = parse_str(tixi.getTextAttribute(node_wing, 'uID'))
        except tixiwrapper.TixiException:
            wing_uid = f'WING{idx_wing:02}'

        logger.debug(f"Loading wing '{wing_uid}'")
        aircraft.add_wing(wing_uid)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        aircraft.wing[wing_uid].symmetry = tigl.wingGetSymmetry(idx_wing)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        node_segments = node_wing + '/segments'

        if not tixi.checkElement(node_segments):
            logger.error(f"Could not find path '{node_segments}'")
            continue

        logger.debug(f"Loading segments of wing '{wing_uid}'...")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # enumerate segments
        for idx_segment in range(1, tixi.getNamedChildrenCount(node_segments, 'segment') + 1):
            node_segment = node_segments + '/segment[{}]'.format(idx_segment)

            try:
                segment_uid = parse_str(tixi.getTextAttribute(node_segment, 'uID'))
            except tixiwrapper.TixiException:
                segment_uid = '{}_SEGMENT{:02}'.format(wing_uid, idx_segment)

            logger.debug("Loading segment '{}'...".format(segment_uid))

            aircraft.wing[wing_uid].add_segment(segment_uid)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            lower = tigl.wingGetLowerPoint(idx_wing, idx_segment, 0.0, 0.0)
            upper = tigl.wingGetUpperPoint(idx_wing, idx_segment, 0.0, 0.0)
            a = [(l + u)/2.0 for l, u in zip(lower, upper)]

            lower = tigl.wingGetLowerPoint(idx_wing, idx_segment, 1.0, 0.0)
            upper = tigl.wingGetUpperPoint(idx_wing, idx_segment, 1.0, 0.0)
            b = [(l + u)/2.0 for l, u in zip(lower, upper)]

            lower = tigl.wingGetLowerPoint(idx_wing, idx_segment, 1.0, 1.0)
            upper = tigl.wingGetUpperPoint(idx_wing, idx_segment, 1.0, 1.0)
            c = [(l + u)/2.0 for l, u in zip(lower, upper)]

            lower = tigl.wingGetLowerPoint(idx_wing, idx_segment, 0.0, 1.0)
            upper = tigl.wingGetUpperPoint(idx_wing, idx_segment, 0.0, 1.0)
            d = [(l + u)/2.0 for l, u in zip(lower, upper)]

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # Re-order vertices
            # * A, D should be at root and B, C at tip
            # * This is done so that the segments (thus panel normals point in the correct direction)
            if b[1] - a[1] < 0.0 or (b[1] == a[1] and b[2] - a[2] < 0.0):
                a, b, c, d = b, a, c, d

            if c[1] - d[1] < 0.0 or (c[1] == d[1] and c[2] - d[2] < 0.0):
                a, b, c, d = a, b, d, c

            if d[0] - a[0] < 0.0:
                a, b, c, d = d, b, c, a

            if c[0] - b[0] < 0.0:
                a, b, c, d = a, c, b, d

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            aircraft.wing[wing_uid].segment[segment_uid].vertices['a'] = a
            aircraft.wing[wing_uid].segment[segment_uid].vertices['b'] = b
            aircraft.wing[wing_uid].segment[segment_uid].vertices['c'] = c
            aircraft.wing[wing_uid].segment[segment_uid].vertices['d'] = d

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            sect, elem = tigl.wingGetInnerSectionAndElementIndex(idx_wing, idx_segment)

            name_ib = parse_str(tigl.wingGetProfileName(idx_wing, sect, elem))
            if not name_ib:
                msg = f"CPACS error: could not extract inner wing profile name (wing: {idx_wing}, segment: {sect})"
                raise ValueError(msg)

            file_ib = os.path.join(settings.dirs['airfoils'], 'blade.{}'.format(name_ib))

            aircraft.wing[wing_uid].segment[segment_uid].airfoils['inner'] = file_ib

            sect, elem = tigl.wingGetOuterSectionAndElementIndex(idx_wing, idx_segment)

            name_ob = parse_str(tigl.wingGetProfileName(idx_wing, sect, elem))
            if not name_ob:
                msg = f"CPACS error: could not extract outer wing profile name (wing: {idx_wing}, segment: {sect})"
                raise ValueError(msg)

            file_ob = os.path.join(settings.dirs['airfoils'], 'blade.{}'.format(name_ob))

            aircraft.wing[wing_uid].segment[segment_uid].airfoils['outer'] = file_ob

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            logger.debug(f"Loaded segment '{segment_uid}'")

        # ===== ADD CONTROLS =====

        # Iterate through component sections (contain control surfaces)
        for idx_comp_section in range(1, tigl.wingGetComponentSegmentCount(idx_wing) + 1):
            name_comp_section = tigl.wingGetComponentSegmentUID(idx_wing, idx_comp_section)

            # Iterate through control surfaces
            for idx_control in range(1, tigl.getControlSurfaceCount(name_comp_section) + 1):
                # Control surfaces can be trailing or leading edge devices
                for device_pos in ('leading', 'trailing'):
                    control_uid = tigl.getControlSurfaceUID(name_comp_section, idx_control)
                    logger.debug("Wing {:d} has control {:s}".format(idx_wing, control_uid))
                    node_control = NODE_CONTROL.format(idx_wing, idx_comp_section, idx_control, device_pos)

                    # Try to read the relative coordinates for each control (eta, xsi)
                    try:
                        # Control surface corner points
                        etaLE_ib = tixi.getDoubleElement(node_control + "/outerShape/innerBorder/etaLE")
                        etaTE_ib = tixi.getDoubleElement(node_control + "/outerShape/innerBorder/etaTE")
                        xsiLE_ib = tixi.getDoubleElement(node_control + "/outerShape/innerBorder/xsiLE")
                        etaLE_ob = tixi.getDoubleElement(node_control + "/outerShape/outerBorder/etaLE")
                        etaTE_ob = tixi.getDoubleElement(node_control + "/outerShape/outerBorder/etaTE")
                        xsiLE_ob = tixi.getDoubleElement(node_control + "/outerShape/outerBorder/xsiLE")

                        # Hinge parameters
                        hingeXsi_ib = tixi.getDoubleElement(node_control + "/path/innerHingePoint/hingeXsi")
                        hingeXsi_ob = tixi.getDoubleElement(node_control + "/path/outerHingePoint/hingeXsi")

                    except tixiwrapper.TixiException:
                        logger.debug("No control data found for NODE {:s}".format(node_control))
                        continue

                    if device_pos == 'leading':
                        # Enforcing parallelism between control edges and x-axis
                        xsiLE_ib = 0.0
                        xsiLE_ob = 0.0

                        # Relative coordinates of control w.r.t. component segment
                        _, segment_uid_inner, eta_inner, xsi_inner, _ = \
                            tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaTE_ib, xsiTE_ib)

                        _, segment_uid_outer, eta_outer, xsi_outer, _ = \
                            tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaTE_ob, xsiTE_ob)

                        # Relative coordinates of control hinge line w.r.t. component segment
                        _, _, _, xsi_h1, _ = tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaTE_ib, hingeXsi_ib)
                        _, _, _, xsi_h2, _ = tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaTE_ob, hingeXsi_ob)

                    elif device_pos == 'trailing':
                        xsiTE_ib = 1.0
                        xsiTE_ob = 1.0

                        # Relative coordinates of control w.r.t. component segment
                        _, segment_uid_inner, eta_inner, xsi_inner, _ = \
                            tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaLE_ib, xsiLE_ib)

                        _, segment_uid_outer, eta_outer, xsi_outer, _ = \
                            tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaLE_ob, xsiLE_ob)

                        # Relative coordinates of control hinge line w.r.t. component segment
                        _, _, _, xsi_h1, _ = tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaLE_ib, hingeXsi_ib)
                        _, _, _, xsi_h2, _ = tigl.wingComponentSegmentPointGetSegmentEtaXsi(name_comp_section, etaLE_ob, hingeXsi_ob)

                    # ADD WING CONTROL AND SET ATTRIBUTES
                    control = aircraft.wing[wing_uid].add_control(control_uid, return_control=True)

                    if device_pos == 'leading':
                        control.device_type = 'slat'
                    elif device_pos == 'trailing':
                        control.device_type = 'flap'

                    # Set DEFAULT deflection to 0
                    control.deflection = 0

                    control.rel_vertices['eta_inner'] = eta_inner
                    control.rel_vertices['xsi_inner'] = xsi_inner
                    control.rel_vertices['eta_outer'] = eta_outer
                    control.rel_vertices['xsi_outer'] = xsi_outer

                    control.rel_hinge_vertices['xsi_inner'] = xsi_h1
                    control.rel_hinge_vertices['xsi_outer'] = xsi_h2

                    control.segment_uid['inner'] = segment_uid_inner
                    control.segment_uid['outer'] = segment_uid_outer

    # ----- CONTROL SURFACE DEFLECTION -----
    try:
        n_control_dev = tixi.getNamedChildrenCount(NODE_TS_CONTROL, 'controlDevice')
    except:
        n_control_dev = 0

    for idx_control in range(1, n_control_dev + 1):
        node_control_device = NODE_TS_CONTROL + '/controlDevice[{}]'.format(idx_control)
        control_uid = tixi.getTextAttribute(node_control_device, 'uID')
        deflection = 0
        deflection_mirror = None

        try:
            deflection = tixi.getDoubleElement(node_control_device + '/deflection')
        except tixiwrapper.TixiException:
            logger.error("Unable to read 'deflection' for control '{:s}'".format(control_uid))

        try:
            deflection_mirror = tixi.getDoubleElement(node_control_device + '/deflectionMirror')
        except:
            logger.warning("Unable to read 'deflection_mirror' for control '{:s}'".format(control_uid))

        deflection_is_set = False

        for this_wing in all_wings(aircraft):
            wing = this_wing[2]

            if control_uid in wing.control.keys():
                wing.control[control_uid].deflection = deflection
                wing.control[control_uid].deflection_mirror = deflection_mirror
                deflection_is_set = True
                break

        if not deflection_is_set:
            logger.error("Could not set deflection for control '{:s}'".format(control_uid))
            raise ComponentDefinitionError("Control '{:s}' not found".format(control_uid))

    # ----- CONTROL CHECKS -----
    for this_control, _ in all_controls(aircraft):
        this_control[2].check()

    # 2.3. SEGMENT WING SECTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    logger.debug("Extracting airfoil data...")
    num_foils = tixi.getNumberOfChilds(NODE_AIRFOILS)

    for i in range(1, num_foils + 1):
        node_airfoil = NODE_AIRFOILS + '/wingAirfoil[{}]'.format(i)
        node_data = node_airfoil + '/pointList'

        try:
            name_airfoil = parse_str(tixi.getTextElement(node_airfoil + '/name'))
        except tixiwrapper.TixiException:
            name_airfoil = f'AIRFOIL{i:02d}'

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        file_airfoil = os.path.join(settings.dirs['airfoils'], 'blade.{}'.format(name_airfoil))

        if os.path.isfile(file_airfoil):
            continue

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        logger.info(f"Copying airfoil {name_airfoil} to file...")

        # coordinate file header
        header = '%{}\n {}'.format(name_airfoil, BOUNDING_BOX)

        # convert string to numpy array
        coords_x = np.fromstring(tixi.getTextElement(node_data + '/x'), sep=';')
        coords_z = np.fromstring(tixi.getTextElement(node_data + '/z'), sep=';')

        coords = np.transpose([coords_x, coords_z])

        np.savetxt(file_airfoil, coords, header=header, fmt=COORD_FORMAT)
        logger.info("airfoil '{}' copied to {}.".format(name_airfoil, file_airfoil))

    logger.debug("Airfoil data extracted...")

    # 2.4. REFERENCE VALUES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    aircraft.refs['gcenter'] = np.zeros(3, dtype=float, order='C')
    aircraft.refs['gcenter'][0] = tixi.getDoubleElement(NODE_REFS + '/point/x')
    aircraft.refs['gcenter'][1] = tixi.getDoubleElement(NODE_REFS + '/point/y')
    aircraft.refs['gcenter'][2] = tixi.getDoubleElement(NODE_REFS + '/point/z')

    # TODO | currently the same as gcenter
    aircraft.refs['rcenter'] = np.zeros(3, dtype=float, order='C')
    aircraft.refs['rcenter'][0] = tixi.getDoubleElement(NODE_REFS + '/point/x')
    aircraft.refs['rcenter'][1] = tixi.getDoubleElement(NODE_REFS + '/point/y')
    aircraft.refs['rcenter'][2] = tixi.getDoubleElement(NODE_REFS + '/point/z')

    # TODO | currently one reference length
    aircraft.refs['area'] = tixi.getDoubleElement(NODE_REFS + '/area')
    aircraft.refs['span'] = tixi.getDoubleElement(NODE_REFS + '/length')
    aircraft.refs['chord'] = tixi.getDoubleElement(NODE_REFS + '/length')

    # 3. GET STATE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # 3.1. GET AERODYNAMIC OPERATING CONDITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # val_alpha = tixi.getTextElement(NODE_AEROPMAP + '/angleOfAttack')
    # val_beta = tixi.getTextElement(NODE_AEROPMAP + '/angleOfYaw')

    # state.aero['alpha'] = np.fromstring(val_alpha, sep=';', dtype=float)
    # state.aero['beta'] = np.fromstring(val_beta, sep=';', dtype=float)
    # TODO | multi-state sweep from aeroPerformanceMap

    # node_states = NODE_TOOLSPEC + '/states'

    # val_airspeed = tixi.getTextElement(node_state + '/alpha')
    # val_alpha = tixi.getTextElement(node_state + '/alpha')
    # val_beta = tixi.getTextElement(node_state + '/beta')
    # val_P = tixi.getTextElement(node_state + '/P')
    # val_Q = tixi.getTextElement(node_state + '/Q')
    # val_R = tixi.getTextElement(node_state + '/R')
    # val_density = tixi.getTextElement(node_state + '/density')

    # state.aero['airspeed'] = np.fromstring(val_airspeed, sep=';', dtype=float)
    # state.aero['alpha'] = np.fromstring(val_alpha, sep=';', dtype=float)
    # state.aero['beta'] = np.fromstring(val_beta, sep=';', dtype=float)
    # state.aero['rate_P'] = np.fromstring(val_P, sep=';', dtype=float)
    # state.aero['rate_Q'] = np.fromstring(val_Q, sep=';', dtype=float)
    # state.aero['rate_R'] = np.fromstring(val_R, sep=';', dtype=float)
    # state.aero['density'] = np.fromstring(val_density, sep=';', dtype=float)
    # TODO | multi-state sweep from toolSpecific

    # state.aero['airspeed'] = tixi.getDoubleElement(node_state + '/airspeed')
    # state.aero['alpha'] = tixi.getDoubleElement(node_state + '/alpha')
    # state.aero['beta'] = tixi.getDoubleElement(node_state + '/beta')
    # state.aero['rate_P'] = tixi.getDoubleElement(node_state + '/P')
    # state.aero['rate_Q'] = tixi.getDoubleElement(node_state + '/Q')
    # state.aero['rate_R'] = tixi.getDoubleElement(node_state + '/R')
    # state.aero['density'] = tixi.getDoubleElement(node_state + '/density')
    # TODO | currently in tool-specific: need altitude, mach

    tixi.save(filepath)
    tixi.close()

    logger.info(f"Aircraft loaded from CPACS file: {filepath}")


##################################################################
##################################################################
# def save_state(state, settings):
#     """
#     Save flight state data to TOOLSPECIFIC in CPACS definition

#     Args:
#         :state: (object) data structure for flight state
#         :settings: (object) data structure for execution settings
#     """

#     # 1. TIXI, TIGL INITIALISATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#     logger.debug("checking TIXI installation...")

#     if not TIXI_INSTALLED:
#         return logger.error("Could not import 'tixiwrapper.py'.")

#     logger.debug("TIXI imported.")
#     logger.debug("checking TIGL installation...")

#     if not TIGL_INSTALLED:
#         return logger.error("could not import 'tiglwrapper.py'.")

#     logger.debug("TIGL imported.")

#     # handles to TIXI, TIGL libraries
#     tixi = tixiwrapper.Tixi()
#     tigl = tiglwrapper.Tigl()

#     filepath = settings.files['aircraft']
#     logger.info(f"Loading aircraft from CPACS file: {filepath}...")

#     if not os.path.exists(filepath):
#         return logger.error("File '{filepath}' not found")

#     tixi.open(filepath)
#     tigl.open(tixi, '')

#     # 2. SET SINGLE STATE IN TOOLSPECIFIC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#     # make node in TOOLSPECIFIC
#     if not tixi.checkElement(NODE_TOOLSPEC):
#         tixi.createElement('/cpacs/toolspecific', 'pyTornado')

#     # make STATES node
#     node_states = NODE_TOOLSPEC + '/states'
#     if not tixi.checkElement(node_states):
#         tixi.createElement(NODE_TOOLSPEC, 'states')

#     # make STATE node with uID STATE.NAME
#     if not tixi.checkElement(node_states + '/state'):
#         tixi.createElement(node_states, 'state')

#     for i in range(1, tixi.getNamedChildrenCount(node_states, 'state') + 1):
#         node_state = node_states + '/state[{}]'.format(i)

#     # if for loop ends without finding entry
#     else:
#         # make STATE node with uID STATE.NAME
#         tixi.createElement(node_states, 'state')

#     if not tixi.checkElement(node_state + '/airspeed'):
#         tixi.addDoubleElement(node_state, 'airspeed', 0.0, '%g')
#     if not tixi.checkElement(node_state + '/alpha'):
#         tixi.addDoubleElement(node_state, 'alpha', 0.0, '%g')
#     if not tixi.checkElement(node_state + '/beta'):
#         tixi.addDoubleElement(node_state, 'beta', 0.0, '%g')
#     if not tixi.checkElement(node_state + '/P'):
#         tixi.addDoubleElement(node_state, 'P', 0.0, '%g')
#     if not tixi.checkElement(node_state + '/Q'):
#         tixi.addDoubleElement(node_state, 'Q', 0.0, '%g')
#     if not tixi.checkElement(node_state + '/R'):
#         tixi.addDoubleElement(node_state, 'R', 0.0, '%g')
#     if not tixi.checkElement(node_state + '/density'):
#         tixi.addDoubleElement(node_state, 'density', 0.0, '%g')

#     # val_airspeed = tixi.getTextElement(node_state + '/alpha')
#     # val_alpha = tixi.getTextElement(node_state + '/alpha')
#     # val_beta = tixi.getTextElement(node_state + '/beta')
#     # val_P = tixi.getTextElement(node_state + '/P')
#     # val_Q = tixi.getTextElement(node_state + '/Q')
#     # val_R = tixi.getTextElement(node_state + '/R')
#     # val_density = tixi.getTextElement(node_state + '/density')

#     # state.aero['airspeed'] = np.fromstring(val_airspeed, sep=';', dtype=float)
#     # state.aero['alpha'] = np.fromstring(val_alpha, sep=';', dtype=float)
#     # state.aero['beta'] = np.fromstring(val_beta, sep=';', dtype=float)
#     # state.aero['rate_P'] = np.fromstring(val_P, sep=';', dtype=float)
#     # state.aero['rate_Q'] = np.fromstring(val_Q, sep=';', dtype=float)
#     # state.aero['rate_R'] = np.fromstring(val_R, sep=';', dtype=float)
#     # state.aero['density'] = np.fromstring(val_density, sep=';', dtype=float)
#     # TODO | multi-state sweep from toolSpecific

#     tixi.updateDoubleElement(node_state + '/airspeed', state.aero['airspeed'], '%g')
#     tixi.updateDoubleElement(node_state + '/alpha', state.aero['alpha'], '%g')
#     tixi.updateDoubleElement(node_state + '/beta', state.aero['beta'], '%g')
#     tixi.updateDoubleElement(node_state + '/P', state.aero['rate_P'], '%g')
#     tixi.updateDoubleElement(node_state + '/Q', state.aero['rate_Q'], '%g')
#     tixi.updateDoubleElement(node_state + '/R', state.aero['rate_R'], '%g')
#     tixi.updateDoubleElement(node_state + '/density', state.aero['density'], '%g')

#     tixi.save(filepath)
#     tixi.close()

#     return logger.info("Flight state written to CPACS file: {}".format(filepath))
##################################################################
##################################################################
