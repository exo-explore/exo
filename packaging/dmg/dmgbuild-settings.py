# Settings for dmgbuild (https://github.com/dmgbuild/dmgbuild), used by
# create-dmg.sh. dmgbuild runs this file with its options as globals, and
# the values given with -D on the command line in `defines`.
#
# Required defines:
#   app         path to EXO.app
#   background  path to background.png (background@2x.png must sit next to it)

import os
from typing import cast

defines = cast(dict[str, str], globals()["defines"])

application = defines["app"]
application_name = os.path.basename(application.rstrip("/"))

# Keep the format the release workflow used before: zlib-compressed, readable
# on every macOS version EXO supports.
format = "UDZO"  # noqa: A001
filesystem = "HFS+"

files = [application]
symlinks = {"Applications": "/Applications"}

# The disk shows EXO's icon when it is mounted.
volume_icon = os.path.join(application, "Contents", "Resources", "AppIcon.icns")
if os.path.isfile(volume_icon):
    icon = volume_icon

background = defines["background"]

# Window position (from the bottom left of the screen) and size, title bar
# included, and the icon centres, in points. Keep the size and centres in
# sync with generate-background.py, which draws the arrow between the icons.
window_rect = ((200, 300), (640, 440))
icon_locations = {
    application_name: (160, 200),
    "Applications": (480, 200),
}

default_view = "icon-view"
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
arrange_by = None
icon_size = 128
text_size = 13
