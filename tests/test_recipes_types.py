from sidequest_daemon.media.recipes import CameraPreset, LOD, PlaceLOD, Slot


def test_slot_has_required_members():
    assert Slot.CASTING.value == "casting"
    assert Slot.LOCATION.value == "location"
    assert Slot.DIRECTION_ACTION.value == "direction_action"
    assert Slot.DIRECTION_CAMERA.value == "direction_camera"
    assert Slot.ART_SENSIBILITY.value == "art_sensibility"


def test_lod_levels():
    assert [m.value for m in LOD] == ["solo", "long", "short", "background"]


def test_place_lod_levels():
    assert [m.value for m in PlaceLOD] == ["solo", "backdrop"]


def test_camera_preset_count_is_seventeen():
    assert len(CameraPreset) == 17


def test_camera_preset_contains_canary_presets():
    assert CameraPreset.portrait_3q in CameraPreset
    assert CameraPreset.topdown_90 in CameraPreset
    assert CameraPreset.extreme_closeup_leone in CameraPreset
    assert CameraPreset.scene in CameraPreset
