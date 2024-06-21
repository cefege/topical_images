from django import forms

class ImageUploadForm(forms.Form):
    frame_image = forms.ImageField(label='Frame Image', required=False)
    pool_image = forms.ImageField(label='Pool Image', required=False)
    title = forms.CharField(max_length=255, required=False)
    input_string = forms.CharField(widget=forms.Textarea, required=False)

