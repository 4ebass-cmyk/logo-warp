# logo-warp

Put a logo onto a product photo so it looks printed on the product, not pasted on top. The mark follows the product's
perspective and curve, bends with fabric folds, and takes the photo's own light and texture.

It is built for merch mockups: t-shirts, hoodies, bags, caps, bottles, mugs, boxes. A typical photo takes 1-2 seconds
and little memory, because only the area around the logo is processed.

## Install

Python 3.10+:

```
pip install -r requirements.txt
```

## Quick start

1. Export the logo as a PNG with a transparent background. The alpha channel is the print.
2. Copy `examples/job-example.json`, then set the photo path and where the logo goes.
3. Run it:

```
python mockup.py my-job.json
```

The result is written to `out`. If `preview` is set, a close crop around the logo is saved too.

## Job settings

| key | what it does |
|---|---|
| `photo`, `logo`, `out` | input photo, logo PNG, output file |
| `center` | `[x, y]` in photo pixels: where the middle of the logo goes |
| `width` | logo width in photo pixels |
| `angle` | tilt of the logo's rows in degrees, clockwise is positive |
| `angle_v` | tilt of the logo's verticals. Leave it out on flat products. On a turned torso the rows and the verticals often differ (e.g. rows 8, verticals 2.5); without it, letters look slanted. |
| `arc` | how many pixels the logo's ends drop (chest, side of a bag, cap front). Negative values lift the ends. |
| `fold` | maximum shift in pixels of the print by fabric folds, taken from the photo's shading. 0 turns it off; 6-14 suits a garment shot at about 4000 px wide. |
| `fold_blur` | smoothness of the fold map in pixels (about 20). Larger values follow only big folds. |
| `ink` | a one-colour print in hex (`"#1c1917"`). Leave it out to keep the logo's own colours. |
| `ink_ref` | optional fabric colour the ink colour was measured against. Without it, the fabric around the logo is used. |
| `light` | 1 = the photo's light and fabric texture show through the ink, 0 = flat ink |
| `opacity` | 1 by default. Real prints are opaque, and in linear light even 3% transparency turns black grey. |
| `soft` | edge softness in pixels (about 0.8), to match the photo's focus |

## Getting it to look right

- Match the ink to a real print on the same product, if there is one. Sample its darkest core colour and the fabric next to it, and use them as `ink` and `ink_ref`.
- Find `angle` and `angle_v` from lines on the product itself: hems, seams, straight edges, knit lines. Do not eyeball them from the logo.
- Start with `fold` at 0, then raise it until the letters follow the folds without breaking up.
- Bake the logo into the photo and use the result as one image. A logo layered on top in a layout tool drifts when someone resizes the photo.

## Other tools in this repo

- `fitplane.py`: exact perspective on a flat face (box, label, device screen). Give it 4 corners of something with a known real
  size on the same face. It fits a homography, checks it against features that must be level, and outputs the logo quad.
- `surfmesh.py`: a general surface map for difficult shapes. It fits a grid to lines that are straight on the real object
  (hems, seams, rims, silhouette edges) plus a prior shape (flat or cylinder), then flattens or warps through it. Use it when
  a simple placement is not enough.

## For AI agents

See `AGENTS.md`.
