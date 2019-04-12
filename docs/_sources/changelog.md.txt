# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.0.0 - 2019-04-12
### Added 
- 'src_tan', 'det_tan', 'src_ort', 'det_ort' can be used as vectors per angle
- display.plot is not changed to plot2d + there is plot3d
- geometry classes: circular, linear, helical

## 0.1.0 - 2019-01-08
### Added
- gradient() and divergence() for TV minimization
- cast2shape() for volume registration
- PyQt viewer in display
- Added det_roll, det_pitch and det_yaw for tilting the detector
- Added read_stack for reading .raw, .mat and other formats.

### Fixed

- Fixed crop / bin for memmaps.

### Removed

- method names in display are made shorter now

## 0.0.1 - 2018-10-24
### Added
- Initial release
