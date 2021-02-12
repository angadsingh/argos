from lib.constants import InputMode

def setup_input_stream(config):
    vs = None
    if config.input_mode == InputMode.RTMP_STREAM:
        from input.rtmpstream import RTMPVideoStream
        vs = RTMPVideoStream(config.rtmp_stream_url).start()
    elif config.input_mode == InputMode.PI_CAM:
        from input.picamstream import PiVideoStream
        kwargs = {}
        if hasattr(config, "picam_resolution"):
            kwargs['picam_resolution'] = config.picam_resolution
        if hasattr(config, "picam_framerate"):
            kwargs['picam_framerate'] = config.picam_framerate
        if hasattr(config, "picam_format"):
            kwargs['picam_format'] = config.picam_format
        vs = PiVideoStream(**kwargs).start()
    elif config.input_mode == InputMode.VIDEO_FILE:
        from input.videofilestream import VideoFileStream
        vs = VideoFileStream(config.video_file_path, config.video_in_sync).start()

    return vs