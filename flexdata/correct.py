import logging
from copy import deepcopy


###############################################################################
#                             Correction profiles                             #
###############################################################################
#
# Once a profile has been added here, it should never be changed to ensure
# reproducibility of existing scripts using the profile.
# If an existing profile needs retroactive corrections, add a new profile
# with a -v2 (-v3, ...) suffix.
# If the physical setup changed, create a new profile with an updated date.

profiles = {
    'cwi-flexray-2022-10-28': {
        'description': """
Correction profile deduced from markers after the October 2022 maintenance.
""",
        'det_tan': 24.0485,
        'src_ort': -5.7730,
        'axs_tan': -0.5010
    },
    'cwi-flexray-2022-05-31': {
        'description': """
correction profile deduced from acquila hc/vc/cor settings after the 31 may 2022 re-calibration. Includes det_roll determined by Robert using markers.
""",
        'det_tan': 24.4203,
        'src_ort': -6.2281,
        'axs_tan': -0.5010,
        'det_roll': -0.262,
    },
    'cwi-flexray-2022-05-31-norotation': {
        'description': """
correction profile deduced from acquila hc/vc/cor settings after the 31 may 2022 re-calibration.
""",
        'det_tan': 24.4203,
        'src_ort': -6.2281,
        'axs_tan': -0.5010,
    },
    'cwi-flexray-2020-03-26': {
        'description': """
Correction profile deduced from Acquila HC/VC/COR settings after the March 2020 maintenance. Includes empirically determined det_roll.
""",
        'det_tan': 24.300,
        'src_ort': -6.086,
        'axs_tan': -0.524,
        'det_roll': -0.175,
    },
    'cwi-flexray-2020-03-26-norotation': {
        'description': """
Correction profile deduced from Acquila HC/VC/COR settings after the March 2020 maintenance.
""",
        'det_tan': 24.300,
        'src_ort': -6.086,
        'axs_tan': -0.524,
    },
    'cwi-flexray-2019-04-24': {
        'description': """
This profile was last updated by Alex Kostenko on 24 April 2019.

It was concurrently updated with the documentation and some other changes in the flexdata codebase.
See:
https://github.com/cicwi/flexDATA/commit/8859bc8073880efcb32cc57b152ef23746993ec1#diff-08f83b989c80a05906f380a66964d8d3L393
""",
        'det_tan': 24,
        'src_ort': -7,
        'axs_tan': -0.5,
    }
}

###############################################################################
#                             Correction functions                            #
###############################################################################


def print_profiles():
    print("Profiles")
    for k, profile in profiles.items():
        print(f" - {k}")
        description = profile.get('description')
        if description is not None:
            for l in description.split('\n'):
                print(f"   {l}")


def correct(geometry, profile=None, do_print_changes=True):
    """Apply a correction profile to projection geometry.


    :param geometry:
    :param profile:
        A string describing the calibration profile that should be applied.
        Use `correct.print_profiles()' to view available profiles.
    :returns:
    :rtype:

    """
    profile_names = "\n".join(profiles.keys())

    prof = profiles.get(profile)
    if prof is None:
        raise ValueError(
            f"Correction profile was not correctly specified. Choose one of:\n"
            f"{profile_names}"
        )

    geometry = deepcopy(geometry)

    for k, v in prof.items():
        # Skip description, it is purely to describe correction profiles.
        if k == 'description':
            continue

        prev = geometry[k]
        if do_print_changes:
            logging.info(f"For {k}: adjusting {prev} by {v}.")
        geometry[k] += v

    geometry.log(f"Applied correction profile '{profile}'")

    return geometry


def correct_roi(geometry):
    """Set the detector ort and tan to be in line with the detector ROI settings

    This function is flexray-specific.

    :param geometry:
    :returns:
    :rtype:

    """
    geometry = deepcopy(geometry)

    # Fix roi:
    roi = geometry.description['roi']

    # ROI is written for a pixel in 1x1 binning, so the centre of the detector
    # is at (767,971), and detector pixel size 74.8 um
    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    detector_pixel_size = 0.0748

    d_ort = centre[1] * detector_pixel_size
    d_tan = centre[0] * detector_pixel_size

    geometry.parameters['det_ort'] += d_ort
    geometry.parameters['det_tan'] += d_tan

    if abs(d_ort) > 1e-6 or abs(d_tan) > 1e-6:
        msg = f"Adjusted detector center to ROI window {roi}: {(d_ort, d_tan)}."
        geometry.log(msg)
        logging.info(msg)

    return geometry


def correct_vol_center(geometry):
    """Move the volume center vertically in-between the source and detector center.

    :param geometry:
    :returns:
    :rtype:

    """

    geometry = deepcopy(geometry)
    vol_tra_z_old = geometry.parameters['vol_tra'][0]
    vol_tra_z_new = (geometry.parameters['det_ort'] * geometry.src2obj +
                   geometry.parameters['src_ort'] * geometry.det2obj) / geometry.src2det
    geometry.parameters['vol_tra'][0] = vol_tra_z_new

    if abs(vol_tra_z_old - vol_tra_z_new) > 1e-6:
        msg = f"Adjusted vertical volume center to source/detector pos: z = {vol_tra_z_new}."
        geometry.log(msg)
        logging.info(msg)

    return geometry
