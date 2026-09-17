from django import forms

ALGORITHM_CHOICES = [
    ("grayscale", "Grayscale Conversion"),
    ("inversion", "Image Inversion"),
    ("gaussian_blur", "Gaussian Blur"),
    ("sobel", "Sobel Edge Detection"),
    ("convolution_naive", "Convolution (Naive CUDA)"),
    ("convolution_tiled", "Convolution (Tiled / Shared Memory CUDA)"),
]

EXECUTION_CHOICES = [
    ("cpu", "CPU"),
    ("cuda", "CUDA"),
    ("both", "Both (compare)"),
]

PRESET_CHOICES = [
    ("sharpen", "Sharpen"),
    ("edge", "Edge Detect"),
    ("emboss", "Emboss"),
    ("box_blur", "Box Blur"),
]


class ImageProcessForm(forms.Form):
    image = forms.ImageField(
        help_text="jpg, jpeg, png, bmp, or webp"
    )
    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES)
    execution = forms.ChoiceField(
        choices=EXECUTION_CHOICES,
        initial="both",
        widget=forms.RadioSelect,
    )

    # Only relevant for gaussian_blur — the template shows/hides this
    # based on the selected algorithm via a small bit of JS, but the field
    # stays optional at the form level regardless, since a mismatched
    # algorithm/field pairing should still validate (the view just ignores
    # fields that don't apply to the chosen algorithm).
    kernel_size = forms.IntegerField(
        required=False, initial=5, min_value=3, max_value=15,
        help_text="Gaussian blur only, must be odd",
    )
    sigma = forms.FloatField(
        required=False, initial=1.4, min_value=0.1,
        help_text="Gaussian blur only",
    )

    # Only relevant for convolution_naive / convolution_tiled.
    preset = forms.ChoiceField(
        required=False, choices=PRESET_CHOICES, initial="sharpen",
        help_text="Convolution only",
    )

    def clean_kernel_size(self):
        value = self.cleaned_data.get("kernel_size")
        if value is not None and value % 2 == 0:
            raise forms.ValidationError("Kernel size must be odd.")
        return value
