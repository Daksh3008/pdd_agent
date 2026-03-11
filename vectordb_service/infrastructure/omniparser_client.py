import replicate


def parse_frame(image_path: str, model_version: str, api_token: str) -> dict:
    """Run OmniParser on a frame image via Replicate."""
    client = replicate.Client(api_token=api_token)
    with open(image_path, "rb") as f:
        output = client.run(model_version, input={"image": f})
    return output
