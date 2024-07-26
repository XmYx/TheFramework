import PIL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch


class ModelManager:
    _models = {}
    @staticmethod
    def get_model(model_name, model_type="text2image"):
        if model_name not in ModelManager._models:
            txt2img_pipeline = StableDiffusionXLPipeline.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
            img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name,
                                                                                unet=txt2img_pipeline.unet,
                                                                                text_encoder=txt2img_pipeline.text_encoder,
                                                                                text_encoder_2=txt2img_pipeline.text_encoder_2,
                                                                                tokenizer=txt2img_pipeline.tokenizer,
                                                                                tokenizer_2=txt2img_pipeline.tokenizer_2,
                                                                                vae=txt2img_pipeline.vae)
            ModelManager._models[model_name] = (txt2img_pipeline, img2img_pipeline,)
        return ModelManager._models[model_name][0] if model_type == "text2image" else ModelManager._models[model_name][1]
class Serializable:

    def __init__(self):
        pass

    def serialize(self, res={}):
        return res

    def deserialize(self, data={}):
        return True
class BaseProcess(Serializable):
    def __init__(self, start_frame=0, end_frame=None, **params):
        super().__init__()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.params = params

    def __call__(self, frame):
        raise NotImplementedError("This method should be implemented by subclasses")

    def is_applicable(self, frame_idx):
        if self.end_frame is None:
            return frame_idx >= self.start_frame
        return self.start_frame <= frame_idx <= self.end_frame


class SDXLGenerateImage(BaseProcess):
    def __call__(self, frame, **kwargs):
        model_name = self.params.get('model_name', 'stabilityai/stable-diffusion-xl-base-1.0')
        prompt = self.params.get('prompt', 'A beautiful landscape')

        pipeline = ModelManager.get_model(model_name, model_type="text2image")
        frame.image = pipeline(prompt, num_inference_steps=5).images[0]
        frame.params['generated'] = True
        return frame


class SDXLImage2Image(BaseProcess):
    def __init__(self, offset=-1, **params):
        super().__init__(**params)
        self.offset = offset
        self.timeline = params.get('timeline')

    def __call__(self, frame, **kwargs):

        model_name = self.params.get('model_name', 'stabilityai/stable-diffusion-xl-base-1.0')
        prompt = self.params.get('prompt', 'A beautiful landscape')
        init_frame_idx = self.timeline.frame_idx + self.offset
        print(init_frame_idx)
        if init_frame_idx < 0 or init_frame_idx >= len(self.timeline.frames):
            raise ValueError("Invalid frame index for initialization image")
        init_image = self.timeline.frames[init_frame_idx].image
        pipeline = ModelManager.get_model(model_name, model_type="image2image")
        frame.image = pipeline(prompt, image=init_image, strength=0.6, num_inference_steps=10).images[0]
        frame.params['image2image'] = True
        return frame
class FrameProcessor:
    def __init__(self, timeline):
        self.timeline = timeline
        self.process_queue = []
        self.intermediate_process_queue = []

    def add_process(self, process):
        self.process_queue.append(process)

    def add_intermediate_process(self, process):
        self.intermediate_process_queue.append(process)

    def process_frame(self, frame_idx):
        frame = self.timeline.frames[frame_idx]
        for process in self.process_queue:
            if process.is_applicable(frame_idx):
                frame = process(frame, timeline=self.timeline, frame_idx=frame_idx)
        print(f"Processed frame {frame_idx}: {frame.params}")
    def process_intermediate(self, frame_idx):
        if frame_idx + 1 < len(self.timeline.frames):
            frame = self.timeline.frames[frame_idx]
            next_frame = self.timeline.frames[frame_idx + 1]
            for process in self.intermediate_process_queue:
                if process.is_applicable(frame_idx):
                    next_frame = process(frame, next_frame, timeline=self.timeline, frame_idx=frame_idx)
            print(f"Applied intermediate process between frame {frame_idx} and {frame_idx + 1}")

    def process_frames(self):
        for i in range(len(self.timeline.frames)):
            self.process_intermediate(i)
            self.process_frame(i)
            self.timeline.frame_idx += 1
        self.timeline.frame_idx = 0
class AbstractTimeline(Serializable):

    frame_idx = 0
    max_frames = 1
    frames = []
    fps = 24

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.processor = FrameProcessor(self)

    def serialize(self, res={}):
        props = {"frame_idx":self.frame_idx,
                 "max_frames":self.max_frames,
                 "frames":[frame.serialize() for frame in self.frames],
                 "fps":self.fps}

        res["abstract_timeline"] = props
        return res

    def deserialize(self, data={}):

        props = data.get("abstract_timeline")

        if props:
            self.frame_idx = props['frame_idx']
            self.max_frames = props['max_frames']
            self.frames = [AbstractFrame().deserialize(frame) for frame in props['frames']]
            self.fps = props['fps']
    def add_process(self, process):
        self.processor.add_process(process)
    def add_intermediate_process(self, process):
        self.processor.add_intermediate_process(process)
    def process_frames(self):
        self.processor.process_frames()


class AbstractFrame(Serializable):

    def __init__(self):
        super().__init__()
        self.image = None
        self.params = {}

    def serialize(self, res={}):
        return res

    def deserialize(self, data={}):
        return True
def main():
    print("Welcome to The Framework!")

    timeline = AbstractTimeline()
    timeline.max_frames = 10
    timeline.frames = [AbstractFrame() for _ in range(timeline.max_frames)]

    # Add processes to the timeline with specific frame ranges
    timeline.add_process(SDXLGenerateImage(start_frame=0, end_frame=0, model_name="stabilityai/stable-diffusion-xl-base-1.0", prompt="A serene mountain landscape"))
    timeline.add_process(SDXLImage2Image(start_frame=1, offset=-1, model_name="stabilityai/stable-diffusion-xl-base-1.0", prompt="A vibrant sunset over the mountains", timeline=timeline))

    # Process frames
    timeline.process_frames()

    # Save the frames as examples
    for idx, frame in enumerate(timeline.frames):
        frame.image.save(f"output_frame_{idx}.png")

if __name__ == "__main__":
    main()