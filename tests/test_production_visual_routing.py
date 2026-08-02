import unittest

from factory.production_visual_routing import route_visual_modes


class ProductionVisualRoutingTests(unittest.TestCase):
    @staticmethod
    def _plan():
        roles = ["hook", "evidence", "mechanism", "comparison", "implication", "cta"]
        return {
            "global_style": "premium editorial",
            "scenes": [
                {
                    "scene_index": index,
                    "role": role,
                    "generation_mode": "image",
                    "image_prompt": "detailed source-grounded image prompt " * 8,
                    "motion_prompt": "controlled motion with stable geometry " * 4,
                }
                for index, role in enumerate(roles)
            ],
        }

    def test_routes_exactly_three_wan_scenes_even_when_model_routes_none(self):
        routed = route_visual_modes(self._plan())
        modes = {
            scene["scene_index"]: scene["generation_mode"]
            for scene in routed["scenes"]
        }

        self.assertEqual(sum(mode == "wan_i2v" for mode in modes.values()), 3)
        self.assertEqual(modes[0], "wan_i2v")
        self.assertEqual(modes[2], "wan_i2v")
        self.assertEqual(modes[3], "wan_i2v")
        self.assertEqual(modes[1], "image")

    def test_routing_is_stable_when_scene_array_is_shuffled(self):
        raw = self._plan()
        raw["scenes"] = list(reversed(raw["scenes"]))

        routed = route_visual_modes(raw)
        selected = {
            scene["scene_index"]
            for scene in routed["scenes"]
            if scene["generation_mode"] == "wan_i2v"
        }

        self.assertEqual(selected, {0, 2, 3})

    def test_router_does_not_mutate_director_response(self):
        raw = self._plan()
        routed = route_visual_modes(raw)

        self.assertIsNot(routed, raw)
        self.assertTrue(all(scene["generation_mode"] == "image" for scene in raw["scenes"]))


if __name__ == "__main__":
    unittest.main()
