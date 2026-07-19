import os
import tempfile
import unittest

from rope.AutoSegments import AutoSegment, mark_segments_for_review
from rope.Automation import AutomationController, JobState, fingerprint_video


class AutomationJobTests(unittest.TestCase):
    def _request(self, directory, video):
        return {
            'video_path': video,
            'output_dir': directory,
            'target_embedding': [0.1, 0.2],
            'source_embedding': [0.3, 0.4],
            'source_labels': ['source.jpg'],
            'video_meta': {'fps': 30.0, 'total_frames': 300},
            'scan_config': {'threshold': 55},
            'render_params': {'SwapperResolution': 128},
        }

    def test_review_gate_is_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, 'input.mp4')
            with open(video, 'wb') as handle:
                handle.write(b'video')
            controller = AutomationController(os.path.join(directory, 'jobs'))
            job = controller.start_job(self._request(directory, video))
            job = controller.begin_scan(job)
            with self.assertRaises(ValueError):
                controller.confirm_segments(
                    job, [{'start_frame': 1, 'end_frame': 2, 'approved': True}], {},
                )
            job = controller.set_scan_results(job, [
                {'start_frame': 1, 'end_frame': 20, 'confidence': 80, 'approved': True},
            ])
            job = controller.confirm_segments(job, job.segments, {'ThreadsSlider': 2})
            self.assertEqual(JobState.RENDERING.value, job.state)
            self.assertTrue(job.user_confirmed)

    def test_pause_resume_and_atomic_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, 'input.mp4')
            with open(video, 'wb') as handle:
                handle.write(b'video')
            root = os.path.join(directory, 'jobs')
            controller = AutomationController(root)
            job = controller.begin_scan(
                controller.start_job(self._request(directory, video))
            )
            job = controller.transition(
                job, JobState.SCANNING,
                scan_checkpoint={'chunk': 2, 'stage': 'refine', 'refine_cursor': 90},
            )
            job = controller.pause_job(job)
            self.assertEqual(JobState.PAUSED.value, job.state)
            reloaded = AutomationController(root).store.load(job.job_id)
            self.assertEqual(90, reloaded.scan_checkpoint['refine_cursor'])
            resumed = AutomationController(root).resume_job(job.job_id)
            self.assertEqual(JobState.SCANNING.value, resumed.state)

    def test_completed_job_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, 'input.mp4')
            with open(video, 'wb') as handle:
                handle.write(b'video')
            controller = AutomationController(os.path.join(directory, 'jobs'))
            job = controller.begin_scan(controller.start_job(self._request(directory, video)))
            job = controller.set_scan_results(job, [
                {'start_frame': 0, 'end_frame': 10, 'confidence': 90, 'approved': True},
            ])
            job = controller.confirm_segments(job, job.segments, {})
            job = controller.transition(job, JobState.MERGING)
            job = controller.transition(job, JobState.MUXING)
            job = controller.transition(job, JobState.QC, final_output='output.mp4')
            controller.complete_job(job, 'output.mp4', {'passed': True})
            self.assertIsNone(controller.store.latest_resumable())

    def test_fingerprint_changes_with_video(self):
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, 'input.mp4')
            with open(video, 'wb') as handle:
                handle.write(b'a')
            first = fingerprint_video(video)
            with open(video, 'ab') as handle:
                handle.write(b'b')
            second = fingerprint_video(video)
            self.assertNotEqual(first['size'], second['size'])

    def test_review_reasons(self):
        segments = [AutoSegment(0, 10, 58, hit_count=1)]
        marked = mark_segments_for_review(segments, threshold=55, fps=30)
        self.assertTrue(marked[0].review_required)
        self.assertEqual(3, len(marked[0].review_reasons))


if __name__ == '__main__':
    unittest.main()
