import logging


###############################################################################
#                             Correction profiles                             #
###############################################################################
profiles = {
    'cwi-flexray-2019-05-24': {
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


def correct(geometry, *, profile=None, do_print_changes=True):
    """Apply a correction profile to projection geometry.


    :param geometry:
    :param profile:
    :returns:
    :rtype:

    """
    profile_names = "\n".join(profiles.keys())

    profile = profiles.get(profile)
    if profile is None:
        raise ValueError(
            "Correction profile was not correctly specified. Choose one of: \n{profile_names}")

    for k, v in profile.items():
        # Skip description, it is purely to describe correction profiles.
        if k == 'description':
            continue

        prev = geometry.get(k)
        if do_print_changes:
            logging.info(f"For {k}: adjusting {prev} by {v}.")
        geometry[k] += v

    # TODO: Add line of logging to geometry to indicate that this
    # profile correction has taken place.

    return geometry


def correct_roi(geometry):
    """Set the detector ort and tan to be in line with the detector ROI settings

    This function is flexray-specific.

    :param geometry:
    :returns:
    :rtype:

    """
    # Fix roi:
    roi = geometry.description['roi']
    # XXX: Why do we hardcode 971 and 767? Does this also work when
    # the geometry has already been binned or hardware binning has
    # been used?
    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]

    # Not sure the binning should be taken into account...
    geometry.parameters['det_ort'] -= centre[1] * \
        geometry.parameters['det_pixel']
    geometry.parameters['det_tan'] -= centre[0] * \
        geometry.parameters['det_pixel']

    return geometry


def correct_vol_center(geometry):
    """Move the volume center in-between the source and detector center.

    :param geometry:
    :returns:
    :rtype:

    """

    geom = geometry
    geom.parameters['vol_tra'][0] = (geom.parameters['det_ort'] * geom.src2obj +
                                     geom.parameters['src_ort'] * geom.det2obj) / geom.src2det

    # TODO: Print changes to volume geometry..

    return geometry
