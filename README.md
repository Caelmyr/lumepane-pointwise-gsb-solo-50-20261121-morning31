# LumePane

LumePane is a local desk for inspecting stained-glass photographs. Catalog a window, compare different lighting and inspection dates, mark visible findings, and preserve the images and versions behind each observation.

## Start

Run `docker compose up -d --build` in this directory and open [the examination desk](http://127.0.0.1:19953/). The service binds to the loopback interface and stores records in the Compose `glass-library` volume. Use `docker compose stop` to stop the service. Starting the same Compose project again reopens the same library.

For a source checkout with Python 3.11 and the packages in `requirements.txt`, run `python3 -m pip install -r requirements.txt` and `./start.sh`. Native source runs default to loopback port 19953 and the `data` directory. Set `PORT` and `LUMEPANE_DATA` to use different local settings. With Compose, set `LUMEPANE_PORT` and choose a separate Compose project name to run an isolated copy.

## Inspect a window

1. In **Window collection**, choose **Load demonstration** and open **Vesper · Thistle roundel**. The photographs and dimensions are synthetic. For your own images, create a window and select **Add photograph**, or use **Photo intake** to review several files and TIFF pages before importing.
2. In **Photographs**, pick the lighting capture. Switch between the source and the registered plane, adjust display levels, and use the original pixel probe for unadjusted values. **Inspect source file** shows preserved file digests, decoding details, and the photographed support through the frame.
3. In **Register**, choose four corners clockwise from the upper left, preview the registered image, and save the registration. Use **Check alignment landmarks** to record independent locations on the window and original image and view the error and support map.
4. In **Markings**, draw a closed region for a glass piece, mask, or reference patch. An open trace can record a fracture or came. The image and the mark record show the appropriate coordinate basis and measurements.
5. In **Examinations**, inspect exposure and readability, measured and overlapping pane regions, inside edges, came strips, fracture candidates, recorded traces, line profiles, texture spectra, transmitted-light ratios, paired-light detail, and photographed color. Select **Preview map** to see the result, or save it to open the evidence images and CSV.
6. In **Compare**, view registered photos with a divider or checkerboard. **Measure visit change** uses captures from the same lighting role and reports areas with enough support for a comparison.
7. In **Overview**, combine several close-up or partial photographs of the same window into one saved whole-window picture. Add each segment from one lighting and one inspection visit, mark the four corners it occupies on the window plane, and preview or save. A clear photographed sample is always kept over an occluded, clipped, or near-dark sample of the same place, so an obstruction in one segment does not erase another segment's clear view. Where two clear segments disagree, both sources stay in the record and a numbered seam marker is drawn; **Trace both sources** reopens the two photographs and their hashes. Places no photograph reached stay empty. Reopen a saved overview, download its CSV, or check its stored media later.
8. In **Visit log**, link dated notes to the versions of observations and examinations you inspected. Create a condition sheet to review the pinned versions together.
9. In **Evidence**, create a contact PDF, a full evidence ZIP, or a scaled detail proof. A received evidence ZIP can be opened in **Received evidence**. Digests verify data in the library. Print scale proofs at 100% and measure the printed 20 mm ruler.

## Image handling

- Accepts PNG, JPEG, TIFF and WebP. A selected frame must be at least 16 pixels on each side and at most 4,500 pixels on either side, 12 megapixels, and 28 MiB. A photograph can be reduced to 1,800 pixels on its longest working edge while the uploaded original is preserved.
- One intake accepts up to eight files and 35 MiB of source bytes. It inspects the first eighteen pages of each staged file and imports up to eighteen selected pages.
- Registration fits a projective plane from four points. It has no model for surface bending or lens distortion. Use registered measurements as image evidence and check the scale and residuals before interpreting a physical dimension.
- The optical reports use working sRGB converted to linear values. Transmitted-light ratios depend on the incident and dark reference and the recorded support. The color and surface reports describe photograph-dependent appearance. They do not measure material chemistry or certify a treatment.
- Unavailable areas, excluded marks, highlights, and low signal can limit specific readings. Any estimate is tied to the photographed areas, chosen scale, and recorded lighting.
- A whole-window overview joins photographs from one lighting and one inspection visit only. It keeps the first clear sample at each place, draws numbered seam markers where clear segments disagree, and leaves unphotographed areas empty. It does not align segments automatically; the operator marks each segment's corners on the window plane.

Repository folders `lumepane/` and `static/` contain the backend and browser interface. The normal backend line count can be reproduced with `python3 scripts/count_backend.py --project-root . --source-root lumepane`.
