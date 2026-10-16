# Instructions for AI agents (Claude Code, Codex, etc.)

You are using this repo to place a client's logo onto a product photo for a mockup. The goal is a mark that looks
physically printed on the product, in its perspective, curvature, folds and light.

## Workflow

1. Look at the photo first and decide the product type:
   - flat: box, label, laptop sleeve, notebook
   - round and rigid: bottle, mug, tumbler, can
   - soft: t-shirt, hoodie, tote, bandana, cap
2. Rasterise the logo to a transparent PNG at roughly 2x the size it will have in the photo. The alpha channel is the print.
3. Read the geometry off the PRODUCT, not off any existing print:
   - Hems, seams, straight edges and rims give the row direction (`angle`) and the curve (`arc`).
   - Knit lines, side seams and the placket give the verticals (`angle_v`). On garments these often differ from the rows.
   - Measure the angles in pixels on a crop. Never guess them.
4. Write a job JSON (see `examples/job-example.json`) and run `python mockup.py job.json`.
5. Open the `preview` crop and check it:
   - Are the letter stems parallel to the product's verticals?
   - Do the rows follow the hem or rim?
   - Is the ink as dark as a real print on similar fabric?
   - Do the folds bend the letters without breaking them?
6. Adjust and rerun. Each run is about 1-2 seconds, so iterate.

## Rules

- If the product already has a real print, use it only to check your result, never to fit the placement. Clients' blank
  products have no print to lean on, so the method has to work without one.
- Keep `opacity` at 1 for printed or embroidered marks, and match `ink` to real printed colours.
- For a flat face with a known-size feature (a screen, a label), use `fitplane.py` to get an exact perspective quad.
- For rigid round products, get the curvature from the product's own circles (rims, neck, lid) and the camera focal
  length from EXIF. A rounded bottom edge exaggerates the curvature, so do not trust it alone.
- Work on crops or downscaled copies when analysing, and keep the full-resolution photo only for the final render.
  The machines running this often have little free RAM.
