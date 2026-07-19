import threading
import time
import unittest

import numpy as np

from rope.FaceTracking import CANONICAL_KPS, TemporalFaceTracker


SLOT = "target-source"
FRAME_SIZE = (1280, 720)
SIGNATURE = np.full((18, 32), 96, dtype=np.uint8)


def landmarks(x=400.0, y=250.0, scale=1.8, angle=0.0):
    centered = CANONICAL_KPS - CANONICAL_KPS.mean(axis=0)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]])
    return (centered @ rotation * scale + np.asarray([x, y])).astype(np.float32)


def observation(kps, similarity=80.0, slot=SLOT):
    return {"kps": np.asarray(kps, dtype=np.float32), "similarities": {slot: similarity}}


def process(tracker, generation, frame, observations, signature=SIGNATURE, slots=None):
    return tracker.process(
        frame, observations, signature, slots=slots or [{"key": SLOT}],
        threshold=55.0, frame_size=FRAME_SIZE, orientation=0.0,
        generation=generation,
    )


class TemporalFaceTrackerTests(unittest.TestCase):
    def test_static_jitter_is_reduced(self):
        rng = np.random.default_rng(1234)
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        raw_centers = []
        filtered_centers = []
        base = landmarks()
        for frame in range(120):
            noisy = base + rng.normal(0.0, 2.0, base.shape)
            result = process(tracker, generation, frame, [observation(noisy)])
            raw_centers.append(noisy.mean(axis=0))
            filtered_centers.append(result[0]["kps"].mean(axis=0))
        raw_rms = np.sqrt(np.square(np.asarray(raw_centers)[20:] - np.mean(raw_centers[20:], axis=0)).mean())
        filtered_rms = np.sqrt(np.square(np.asarray(filtered_centers)[20:] - np.mean(filtered_centers[20:], axis=0)).mean())
        self.assertLess(filtered_rms, raw_rms * 0.60)

    def test_linear_motion_has_bounded_lag(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        lag = []
        face_scale = np.linalg.norm(landmarks()[0] - landmarks()[4])
        for frame in range(90):
            raw = landmarks(x=400.0 + frame * 0.8)
            result = process(tracker, generation, frame, [observation(raw)])
            lag.append(abs(result[0]["kps"].mean(axis=0)[0] - raw.mean(axis=0)[0]))
        self.assertLess(np.mean(lag[20:]), face_scale * 0.03)

    def test_angle_wrap_does_not_flip(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        centers = []
        for frame, angle in enumerate(np.linspace(3.0, 3.3, 20)):
            result = process(tracker, generation, frame, [observation(landmarks(angle=angle))])
            centers.append(result[0]["kps"])
        jumps = [np.linalg.norm(centers[i] - centers[i - 1], axis=1).mean() for i in range(1, len(centers))]
        self.assertLess(max(jumps), 10.0)

    def test_predicts_exactly_one_detector_miss(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        self.assertTrue(process(tracker, generation, 0, [observation(landmarks())]))
        self.assertTrue(process(tracker, generation, 1, [observation(landmarks(x=401.0))]))
        predicted = process(tracker, generation, 2, [])
        self.assertEqual(1, len(predicted))
        self.assertTrue(predicted[0]["predicted"])
        self.assertEqual([], process(tracker, generation, 3, []))

    def test_scene_cut_prevents_prediction(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        process(tracker, generation, 0, [observation(landmarks())])
        process(tracker, generation, 1, [observation(landmarks())])
        cut = np.full((18, 32), 255, dtype=np.uint8)
        self.assertEqual([], process(tracker, generation, 2, [], signature=cut))

    def test_active_track_uses_small_identity_hysteresis(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        self.assertTrue(process(tracker, generation, 0, [observation(landmarks(), 70.0)]))
        # Below the acquisition threshold (55), but inside threshold - 3 and
        # spatially continuous with the confirmed track.
        self.assertTrue(process(tracker, generation, 1, [observation(landmarks(x=401), 53.0)]))

    def test_large_motion_resets_instead_of_coasting(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        process(tracker, generation, 0, [observation(landmarks())])
        process(tracker, generation, 1, [observation(landmarks(x=401))])
        # A real detection exists but is far outside the motion gate. It must
        # neither inherit the active track nor trigger a predicted paste.
        self.assertEqual([], process(
            tracker, generation, 2, [observation(landmarks(x=900))]
        ))
        # With the old track removed, the next frame can acquire afresh.
        self.assertTrue(process(
            tracker, generation, 3, [observation(landmarks(x=901))]
        ))

    def test_assignment_change_drops_hysteresis(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        process(tracker, generation, 0, [observation(landmarks(), 70.0)])
        changed_slot = "replacement-source"
        changed_observation = {
            "kps": landmarks(x=401),
            "similarities": {changed_slot: 53.0},
        }
        result = tracker.process(
            1, [changed_observation], SIGNATURE,
            slots=[{"key": changed_slot}], threshold=55,
            frame_size=FRAME_SIZE, generation=generation,
        )
        self.assertEqual([], result)

    def test_matching_is_one_to_one(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        second_slot = "second"
        observations = [
            {"kps": landmarks(), "similarities": {SLOT: 90.0, second_slot: 89.0}},
            {"kps": landmarks(x=700), "similarities": {SLOT: 88.0, second_slot: 87.0}},
        ]
        result = tracker.process(
            0, observations, SIGNATURE,
            slots=[{"key": SLOT}, {"key": second_slot}], threshold=55,
            frame_size=FRAME_SIZE, generation=generation,
        )
        self.assertEqual(2, len(result))
        self.assertEqual(2, len({item["slot_key"] for item in result}))
        self.assertEqual(2, len({item["observation_index"] for item in result}))

    def test_checkpoint_round_trip_matches_continuous_output(self):
        continuous = TemporalFaceTracker()
        generation = continuous.start_sequence(0, 30.0)
        for frame in range(10):
            process(continuous, generation, frame, [observation(landmarks(x=400 + frame))])
        checkpoint = continuous.export_checkpoint()
        expected = process(continuous, generation, 10, [observation(landmarks(x=410))])[0]["kps"]

        resumed = TemporalFaceTracker()
        resumed_generation = resumed.start_sequence(10, 30.0, checkpoint)
        actual = process(resumed, resumed_generation, 10, [observation(landmarks(x=410))])[0]["kps"]
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_out_of_order_workers_are_processed_in_frame_order(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        completed = []

        def run(frame):
            result = process(tracker, generation, frame, [observation(landmarks(x=400 + frame))])
            completed.append((frame, result[0]["kps"].mean(axis=0)[0]))

        later = threading.Thread(target=run, args=(1,))
        later.start()
        time.sleep(0.05)
        first = threading.Thread(target=run, args=(0,))
        first.start()
        first.join(timeout=2)
        later.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertFalse(later.is_alive())
        self.assertEqual([0, 1], [item[0] for item in completed])

    def test_abort_releases_waiting_worker(self):
        tracker = TemporalFaceTracker()
        generation = tracker.start_sequence(0, 30.0)
        result = []

        waiting = threading.Thread(
            target=lambda: result.extend(process(
                tracker, generation, 1, [observation(landmarks())]
            ))
        )
        waiting.start()
        time.sleep(0.05)
        tracker.abort_sequence(generation)
        waiting.join(timeout=2)
        self.assertFalse(waiting.is_alive())
        self.assertEqual([], result)


if __name__ == "__main__":
    unittest.main()
