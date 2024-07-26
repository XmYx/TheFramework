from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Import the existing classes and functions
from main import AbstractTimeline, AbstractFrame, SDXLGenerateImage, SDXLImage2Image

class ProcessRegistry:
    _processes = {}

    @staticmethod
    def register_process(name, process_cls):
        ProcessRegistry._processes[name] = process_cls

    @staticmethod
    def get_process(name, **params):
        process_cls = ProcessRegistry._processes.get(name)
        if not process_cls:
            raise ValueError(f"Process {name} is not registered.")
        return process_cls(**params)

    @staticmethod
    def get_all_processes():
        return list(ProcessRegistry._processes.keys())

# Register the initial processes
ProcessRegistry.register_process('SDXLGenerateImage', SDXLGenerateImage)
ProcessRegistry.register_process('SDXLImage2Image', SDXLImage2Image)

timeline = AbstractTimeline()
timeline.max_frames = 10
timeline.frames = [AbstractFrame() for _ in range(timeline.max_frames)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_frame', methods=['POST'])
def add_frame():
    frame_data = request.json
    new_frame = AbstractFrame()
    # You can set additional parameters for the frame if needed
    timeline.frames.append(new_frame)
    return jsonify({"status": "success", "message": "Frame added", "frame_id": len(timeline.frames) - 1})

@app.route('/add_process', methods=['POST'])
def add_process():
    process_data = request.json
    process_type = process_data['type']
    process_params = process_data['params']
    process = ProcessRegistry.get_process(process_type, **process_params)
    timeline.add_process(process)
    return jsonify({"status": "success", "message": "Process added"})

@app.route('/process_frames', methods=['POST'])
def process_frames():
    timeline.process_frames()
    return jsonify({"status": "success", "message": "Frames processed"})

@app.route('/get_frame/<int:frame_id>')
def get_frame(frame_id):
    frame = timeline.frames[frame_id]
    # Save frame image to a temporary file and send it back
    frame.image.save(f"static/frame_{frame_id}.png")
    return jsonify({"status": "success", "image_url": f"/static/frame_{frame_id}.png", "params": frame.params})

@app.route('/available_processes')
def available_processes():
    processes = ProcessRegistry.get_all_processes()
    return jsonify({"processes": processes})

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
