# Background Removal Tool Blueprint

This tool handles removing the background from a source product image or portrait, instructing the AI/Tool pipeline to explicitly perform a 'Composite Layer' operation to place the subject on a solid background color canvas.

Key Phrases: background removal, remove background, transparent, cutout, rmbg, background colors, composite layer.

### Processing Logic:
- Step 1: Perform background subtraction.
- Step 2: Create a solid RGB(255, 255, 255) background layer.
- Step 3: Composite the subject onto the white background layer using alpha blending.
- Step 4: Return the result as a flattened JPEG.

### Configurable Variables:
- `$$Background Color`: Configurable variable for the background color (defaults to `#FFFFFF`).

```json
{
  "tool_id": "bg_remover",
  "show_upload": true,
  "show_url_input": true,
  "layout_mode": "interactive",
  "config": {
    "api_provider": "remove.bg",
    "output_format": "jpeg",
    "active_tool": "bg_remover",
    "default_background": "#FFFFFF",
    "variables": {
      "background_color": "$$Background Color"
    }
  }
}
```
