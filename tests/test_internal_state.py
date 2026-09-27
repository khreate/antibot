import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graph_memory import AssociativeGraph
from mental_state import MentalState


class AssociativeGraphTests(unittest.TestCase):
    def test_activation_spreads_across_configured_hops(self):
        graph = AssociativeGraph(
            edge_half_life_s=10**9,
            activation_half_life_s=10**9,
            spread_hops=2,
        )
        graph.touch_edge("node:a", "node:b", boost=1.0)
        graph.touch_edge("node:b", "node:c", boost=1.0)

        graph.activate("node:a", amount=1.0)

        self.assertGreater(graph.warmth("node:b"), 0.0)
        self.assertGreater(graph.warmth("node:c"), 0.0)

    def test_topic_seed_is_channel_and_scope_isolated(self):
        graph = AssociativeGraph(edge_half_life_s=10**9, activation_half_life_s=10**9)
        guild_one_channel = "channel:guild:1:10"
        guild_two_channel = "channel:guild:2:20"
        guild_one_topic = "topic:guild:1:apples"
        guild_two_topic = "topic:guild:2:bananas"
        graph.record_interaction("user:guild:1:100", [guild_one_topic], guild_one_channel)
        graph.record_interaction("user:guild:2:200", [guild_two_topic], guild_two_channel)

        first_seed = graph.something_on_your_mind(
            channel_id=guild_one_channel, min_warmth=0.0, quiet_for_s=0
        )
        second_seed = graph.something_on_your_mind(
            channel_id=guild_two_channel, min_warmth=0.0, quiet_for_s=0
        )

        self.assertEqual(first_seed, guild_one_topic)
        self.assertEqual(second_seed, guild_two_topic)

    def test_graph_round_trip_preserves_associations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "state.db")
            graph = AssociativeGraph(db_path=db_path)
            graph.record_interaction(
                "user:guild:1:100",
                ["topic:guild:1:python"],
                "channel:guild:1:10",
            )
            graph.save()

            restored = AssociativeGraph(db_path=db_path)
            restored.load()

            self.assertGreater(
                restored.association("channel:guild:1:10", "topic:guild:1:python"),
                0.0,
            )


class HierarchicalBoredomTests(unittest.TestCase):
    def _state(self, now):
        with patch("mental_state.time.time", return_value=now):
            return MentalState(
                seed_traits={
                    "boredom_rate": 0.001,
                    "startup_grace_s": 0,
                    "energy_sigma": 0.0,
                    "valence_sigma": 0.0,
                    "social_battery_sigma": 0.0,
                    "curiosity_sigma": 0.0,
                }
            )

    def test_activity_resets_guild_but_not_other_guilds(self):
        state = self._state(0.0)
        with patch("mental_state.time.time", return_value=1000.0):
            state.on_message(channel_id=10, scope_id="guild:1")
        with patch("mental_state.time.time", return_value=1100.0):
            self.assertAlmostEqual(state.scope_boredom("guild:1"), 0.1)
            self.assertEqual(state.scope_boredom("guild:2"), 1.0)
            self.assertEqual(state.channel_boredom(20), 1.0)

    def test_direct_messages_have_independent_boredom(self):
        state = self._state(0.0)
        with patch("mental_state.time.time", return_value=900.0):
            state.on_message(channel_id=30, scope_id="dm:30")
        with patch("mental_state.time.time", return_value=1000.0):
            self.assertAlmostEqual(state.scope_boredom("dm:30"), 0.1)
            self.assertEqual(state.scope_boredom("dm:40"), 1.0)

    def test_bot_speaking_settles_entire_guild(self):
        state = self._state(0.0)
        with patch("mental_state.time.time", return_value=1000.0):
            self.assertEqual(state.scope_boredom("guild:1"), 1.0)
            state.on_bot_spoke(channel_id=10, scope_id="guild:1", unprompted=True)
            self.assertEqual(state.scope_boredom("guild:1"), 0.0)
            self.assertEqual(state.channel_boredom(10), 0.0)

    def test_load_does_not_count_offline_time_as_boredom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "state.db")
            with patch("mental_state.time.time", return_value=1000.0):
                state = MentalState(db_path=db_path, seed_traits={"boredom_rate": 0.001})
                state.on_message(channel_id=10, scope_id="guild:1")
                state.save()

            with patch("mental_state.time.time", return_value=5000.0):
                restored = MentalState(db_path=db_path, seed_traits={"boredom_rate": 0.001})
                restored.load()
                self.assertEqual(restored.scope_boredom("guild:1"), 0.0)
                self.assertEqual(restored.channel_boredom(10), 0.0)


if __name__ == "__main__":
    unittest.main()
