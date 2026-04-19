import json
path = r'c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\.ipynb_checkpoints\Fake_certificate_detection-checkpoint.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if 'source' in cell:
        for i, line in enumerate(cell['source']):
            if '"valid"' in line and '"val"' in line:
                cell['source'][i] = line.replace('"valid"', r'"d1Certificate forgery detection/valid"')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook updated successfully!')
