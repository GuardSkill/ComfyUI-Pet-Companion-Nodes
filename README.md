# ComfyUI Pet Companion Nodes

Production-oriented green-screen and transparent-loop utilities for real-pet companion animations.

## Nodes

### Pet / Normalize Green First Frame

Replaces only background regions connected to the canvas boundary with an exact chroma color. This helps retain dark noses, pupils, coat markings, and interior colors while cleaning irregular generated backdrops and negative spaces around separated legs.

### Pet / Chroma Key + Closed Loop WebP

Processes an `IMAGE` batch as a video sequence:

- edge-connected chroma keying;
- soft matte generation;
- local green-spill suppression;
- optional exact endpoint closure by replacing the last frame with the first;
- lossless animated RGBA WebP export;
- checkerboard preview, foreground mask, and saved output path.

## Installation

Install from ComfyUI Manager/Registry, or clone into `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/GuardSkill/ComfyUI-Pet-Companion-Nodes.git
```

Restart ComfyUI after installation. The nodes appear under `Pet Companion/Green Screen`.

## Typical workflow

```text
Pet photo
  -> identity-preserving image edit on solid #00FF00
  -> Pet / Normalize Green First Frame
  -> image-to-video model
  -> Pet / Chroma Key + Closed Loop WebP
```

Recommended starting values for H3 green-screen output at 24 fps:

- key color: `#00FF00`
- similarity: `0.20`
- smoothness: `0.10`
- despill: `0.85`
- close loop: enabled
- lossless: enabled

## Notes

- The keyer intentionally starts from regions connected to the frame boundary, reducing accidental removal of similar colors inside the subject.
- Exact closure changes only the final exported frame; interior generated frames are retained.
- The node writes animated WebP files under the configured ComfyUI output directory.

## License

MIT
