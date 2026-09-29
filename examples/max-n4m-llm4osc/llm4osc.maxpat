{
  "patcher": {
    "fileversion": 1,
    "appversion": {
      "major": 8,
      "minor": 6,
      "revision": 0,
      "architecture": "x64",
      "modernui": 1
    },
    "classnamespace": "box",
    "rect": [80.0, 80.0, 920.0, 620.0],
    "bglocked": 0,
    "openinpresentation": 0,
    "default_fontname": "Arial",
    "default_fontsize": 12.0,
    "default_fontface": 0,
    "gridonopen": 1,
    "gridsize": [15.0, 15.0],
    "gridsnaponopen": 1,
    "objectsnaponopen": 1,
    "statusbarvisible": 2,
    "toolbarvisible": 1,
    "lefttoolbarpinned": 0,
    "toptoolbarpinned": 0,
    "righttoolbarpinned": 0,
    "bottomtoolbarpinned": 0,
    "toolbars_unpinned_last_save": 0,
    "tallnewobj": 0,
    "boxanimatetime": 200,
    "enablehscroll": 1,
    "enablevscroll": 1,
    "devicewidth": 0.0,
    "description": "LLM4OSC Node for Max example — resolve NL via local serve; UDP only if live=1",
    "digest": "LLM4OSC N4M thin client",
    "tags": "llm4osc, osc, n4m",
    "style": "",
    "subpatcher_template": "",
    "assistshowspatchername": 0,
    "boxes": [
      {
        "box": {
          "id": "obj-title",
          "maxclass": "comment",
          "text": "LLM4OSC · Node for Max example (thin client → llm4osc serve)",
          "fontsize": 14.0,
          "fontface": 1,
          "patching_rect": [30.0, 20.0, 520.0, 24.0]
        }
      },
      {
        "box": {
          "id": "obj-help",
          "maxclass": "comment",
          "text": "1) llm4osc serve --no-preload   2) script start   3) resolve / example chips   4) live=0 keeps UDP off",
          "patching_rect": [30.0, 48.0, 700.0, 20.0]
        }
      },
      {
        "box": {
          "id": "obj-node",
          "maxclass": "newobj",
          "text": "node.script llm4osc.js @autostart 0 @watch 1",
          "outlettype": [""],
          "patching_rect": [30.0, 200.0, 280.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-start",
          "maxclass": "message",
          "text": "script start",
          "patching_rect": [30.0, 90.0, 80.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-stop",
          "maxclass": "message",
          "text": "script stop",
          "patching_rect": [120.0, 90.0, 75.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-health",
          "maxclass": "message",
          "text": "health",
          "patching_rect": [210.0, 90.0, 50.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-backend",
          "maxclass": "message",
          "text": "backend b0",
          "patching_rect": [280.0, 90.0, 80.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-nl",
          "maxclass": "textedit",
          "lines": 2,
          "fontsize": 12.0,
          "patching_rect": [30.0, 130.0, 360.0, 40.0],
          "parameter_enable": 0
        }
      },
      {
        "box": {
          "id": "obj-nl-label",
          "maxclass": "comment",
          "text": "NL (textedit → prepend resolve)",
          "patching_rect": [30.0, 112.0, 200.0, 18.0]
        }
      },
      {
        "box": {
          "id": "obj-prepend",
          "maxclass": "newobj",
          "text": "prepend resolve",
          "patching_rect": [30.0, 175.0, 100.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-ex1",
          "maxclass": "message",
          "text": "resolve set gain to 50%",
          "patching_rect": [420.0, 90.0, 160.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-ex2",
          "maxclass": "message",
          "text": "resolve make the level half",
          "patching_rect": [420.0, 120.0, 170.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-ex3",
          "maxclass": "message",
          "text": "resolve boost the bass band by 3db",
          "patching_rect": [420.0, 150.0, 220.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-ex4",
          "maxclass": "message",
          "text": "resolve start",
          "patching_rect": [420.0, 180.0, 90.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-route",
          "maxclass": "newobj",
          "text": "route osc refuse error health status",
          "outlettype": ["", "", "", "", "", ""],
          "patching_rect": [30.0, 240.0, 280.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-osc-print",
          "maxclass": "newobj",
          "text": "print llm4osc-osc",
          "patching_rect": [30.0, 280.0, 110.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-refuse-print",
          "maxclass": "newobj",
          "text": "print llm4osc-refuse",
          "patching_rect": [150.0, 280.0, 120.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-error-print",
          "maxclass": "newobj",
          "text": "print llm4osc-error",
          "patching_rect": [290.0, 280.0, 110.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-gate",
          "maxclass": "newobj",
          "text": "gate",
          "patching_rect": [30.0, 320.0, 40.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-live",
          "maxclass": "toggle",
          "patching_rect": [30.0, 290.0, 24.0, 24.0]
        }
      },
      {
        "box": {
          "id": "obj-live-label",
          "maxclass": "comment",
          "text": "live send (off = preview only)",
          "patching_rect": [60.0, 292.0, 180.0, 20.0]
        }
      },
      {
        "box": {
          "id": "obj-udpsend",
          "maxclass": "newobj",
          "text": "udpsend 127.0.0.1 7400",
          "patching_rect": [30.0, 360.0, 150.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-udp-note",
          "maxclass": "comment",
          "text": "Target another Max patch with [udpreceive 7400] + route for /gain /volume /freq …",
          "patching_rect": [30.0, 390.0, 520.0, 20.0]
        }
      },
      {
        "box": {
          "id": "obj-panel",
          "maxclass": "newobj",
          "text": "print llm4osc-status",
          "patching_rect": [420.0, 240.0, 120.0, 22.0]
        }
      }
    ],
    "lines": [
      {
        "patchline": {
          "source": ["obj-start", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-stop", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-health", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-backend", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-nl", 0],
          "destination": ["obj-prepend", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-prepend", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-ex1", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-ex2", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-ex3", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-ex4", 0],
          "destination": ["obj-node", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-node", 0],
          "destination": ["obj-route", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-route", 0],
          "destination": ["obj-osc-print", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-route", 0],
          "destination": ["obj-gate", 1]
        }
      },
      {
        "patchline": {
          "source": ["obj-route", 1],
          "destination": ["obj-refuse-print", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-route", 2],
          "destination": ["obj-error-print", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-route", 3],
          "destination": ["obj-panel", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-route", 4],
          "destination": ["obj-panel", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-live", 0],
          "destination": ["obj-gate", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-gate", 0],
          "destination": ["obj-udpsend", 0]
        }
      }
    ]
  }
}
