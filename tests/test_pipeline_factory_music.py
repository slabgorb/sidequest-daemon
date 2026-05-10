import asyncio
from sidequest_daemon.media.pipeline_factory import MediaPipelineFactory
from sidequest_daemon.media.music_pipeline import MusicPipeline


def test_factory_constructs_music_pipeline():
    factory = MediaPipelineFactory()
    factory.init_music(render_lock=asyncio.Lock())
    assert isinstance(factory.music_pipeline, MusicPipeline)
