import unittest
import tempfile
import numpy as np

from rope.AutoSegments import (
    AutoSegment, approved_ranges, frame_in_ranges, group_matches,
    load_scan_cache, save_scan_cache, scan_cache_key,
    load_scan_manifest, save_scan_manifest,
)


class AutoSegmentsTests(unittest.TestCase):
    def test_groups_hits_and_clamps_padding(self):
        segments = group_matches(
            [(0, 80), (10, 90), (70, 75)], stride=10,
            gap_frames=10, padding_frames=5, total_frames=80,
        )
        self.assertEqual([(0, 24), (65, 79)], [
            (item.start_frame, item.end_frame) for item in segments
        ])
        self.assertEqual(85.0, segments[0].confidence)

    def test_review_filters_rejected_ranges(self):
        segments = [AutoSegment(1, 5, 90, True), AutoSegment(8, 10, 80, False)]
        self.assertEqual([(1, 5)], approved_ranges(segments))
        self.assertTrue(frame_in_ranges(3, approved_ranges(segments)))
        self.assertFalse(frame_in_ranges(8, approved_ranges(segments)))

    def test_empty_input(self):
        self.assertEqual([], group_matches(
            [], stride=10, gap_frames=5, padding_frames=2, total_frames=100,
        ))

    def test_scan_cache_round_trip_and_video_change_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            video = directory + '/sample.mp4'
            with open(video, 'wb') as handle:
                handle.write(b'video-a')
            embedding = np.array([0.1, 0.2], dtype=np.float32)
            key = scan_cache_key(video, embedding, {'stride': 10})
            save_scan_cache(directory, key, [AutoSegment(2, 8, 88)])
            loaded = load_scan_cache(directory, key)
            self.assertEqual((2, 8), (loaded[0].start_frame, loaded[0].end_frame))
            with open(video, 'ab') as handle:
                handle.write(b'-changed')
            self.assertNotEqual(key, scan_cache_key(video, embedding, {'stride': 10}))

    def test_chunk_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {
                'completed_chunks': [0, 1],
                'chunks': {
                    '0': {'hits': [[149, 91.0]], 'near_hits': []},
                    '1': {'hits': [[152, 92.0]], 'near_hits': [[160, 69.0]]},
                },
            }
            save_scan_manifest(directory, 'job', manifest)
            loaded = load_scan_manifest(directory, 'job')
            self.assertEqual([0, 1], loaded['completed_chunks'])
            hits = [tuple(item) for chunk in loaded['chunks'].values()
                    for item in chunk['hits']]
            segments = group_matches(
                hits, stride=3, gap_frames=2,
                padding_frames=0, total_frames=300,
            )
            self.assertEqual([(149, 154)], [
                (item.start_frame, item.end_frame) for item in segments
            ])


if __name__ == '__main__':
    unittest.main()
