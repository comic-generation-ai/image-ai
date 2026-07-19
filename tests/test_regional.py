from core.pipeline_runner import PipelineRunner, ImageRequest

runner = PipelineRunner()
resp = runner.generate(ImageRequest(
    prompt="on the left, a green turtle with a small brown shell, racing forward; "
           "on the right, a white rabbit with long ears, racing forward, forest dirt path, side view",
    width=1024, height=1024, steps=25,
))
resp.image.save("/tmp/regional_test.png")
print("seed used:", resp.seed)