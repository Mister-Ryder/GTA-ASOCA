from pathlib import Path
import csv
import json
import numpy as np
import SimpleITK as sitk

ORIGINAL_SOURCE = Path(r'E:\00-Data\冠状动脉-ASOCA\CompressedVersion')
SOURCE = Path(r'E:\00-Data\ASOCA_AUDIT_ASCII')
OUTPUT = Path(__file__).resolve().parent

def header(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return {'size_xyz': list(reader.GetSize()), 'spacing_xyz_mm': list(reader.GetSpacing()),
            'origin_xyz_mm': list(reader.GetOrigin()), 'direction': list(reader.GetDirection()),
            'pixel_type': sitk.GetPixelIDValueAsString(reader.GetPixelID()), 'file_bytes': path.stat().st_size}

annotated = []
blind = []
errors = []
for group in ['Normal', 'Diseased']:
    group_path = SOURCE / group
    for annotation in sorted((group_path / 'Annotations').glob('*.nrrd')):
        image_path = group_path / 'CTCA' / annotation.name
        image_header = header(image_path)
        label_header = header(annotation)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(annotation)))
        labels, counts = np.unique(mask, return_counts=True)
        matching = all(np.allclose(image_header[field], label_header[field], atol=1e-6) for field in ['size_xyz', 'spacing_xyz_mm', 'origin_xyz_mm', 'direction'])
        if not matching:
            errors.append(annotation.name + ': image/annotation physical geometry differs')
        annotated.append({'case_id': annotation.stem, 'group': group, 'original_partition': 'challenge_training',
                          'image_relative_path': image_path.relative_to(SOURCE).as_posix(),
                          'label_relative_path': annotation.relative_to(SOURCE).as_posix(),
                          'labels': {str(v.item()): int(n) for v,n in zip(labels, counts)},
                          'foreground_fraction': float(np.count_nonzero(mask) / mask.size),
                          'geometry_matches': matching, 'image_header': image_header,
                          'annotation_header': label_header})
        print(f'Audited {annotation.stem}: labels {labels.tolist()}', flush=True)
        del mask
    blind_folder = 'Testset_Normal' if group == 'Normal' else 'Testset_Disease'
    for image_path in sorted((group_path / blind_folder).glob('*.nrrd')):
        blind.append({'case_id': f'{group}_blind_{image_path.stem}', 'group': group,
                      'original_partition': 'challenge_blind_test_unlabeled',
                      'image_relative_path': image_path.relative_to(SOURCE).as_posix(),
                      'label_relative_path': None, 'image_header': header(image_path)})

files = [p for p in SOURCE.rglob('*') if p.is_file()]
image_bytes = sum(row['image_header']['file_bytes'] for row in annotated + blind)
annotation_bytes = sum(row['annotation_header']['file_bytes'] for row in annotated)
foreground = [r['foreground_fraction'] for r in annotated]
spacings = np.array([r['image_header']['spacing_xyz_mm'] for r in annotated])
sizes = np.array([r['image_header']['size_xyz'] for r in annotated])
summary = {'source': str(ORIGINAL_SOURCE), 'reader_ascii_junction': str(SOURCE), 'audit_date': '2026-10-13',
           'annotated_case_count': len(annotated), 'annotated_group_counts': {g: sum(r['group']==g for r in annotated) for g in ['Normal','Diseased']},
           'blind_unlabeled_case_count': len(blind), 'blind_group_counts': {g: sum(r['group']==g for r in blind) for g in ['Normal','Diseased']},
           'all_annotation_labels': sorted({int(v) for r in annotated for v in r['labels']}),
           'labels_identical_across_cases': all(sorted(map(int,r['labels'])) == [0,1] for r in annotated),
           'all_pair_geometries_match': all(r['geometry_matches'] for r in annotated),
           'annotated_size_xyz_min': sizes.min(axis=0).tolist(), 'annotated_size_xyz_max': sizes.max(axis=0).tolist(),
           'annotated_spacing_xyz_mm_min': spacings.min(axis=0).tolist(), 'annotated_spacing_xyz_mm_max': spacings.max(axis=0).tolist(),
           'annotated_foreground_fraction_min': min(foreground), 'annotated_foreground_fraction_max': max(foreground),
           'annotated_foreground_fraction_mean': float(np.mean(foreground)),
           'file_count': len(files), 'total_source_bytes': sum(p.stat().st_size for p in files),
           'all_ctca_bytes': image_bytes, 'all_annotation_bytes': annotation_bytes,
           'annotated_training_pair_bytes': sum(r['image_header']['file_bytes'] + r['annotation_header']['file_bytes'] for r in annotated),
           'errors': errors,
           'scope': 'Label and geometry integrity only. Blind images read at header level; no test annotations or performance-based tuning used.'}
manifest = {'summary': summary, 'annotated_cases': annotated, 'blind_test_cases': blind}
(OUTPUT / 'asoca_dataset_audit.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUTPUT / 'asoca_case_inventory.csv').open('w', encoding='utf-8-sig', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=['case_id','group','original_partition','image_relative_path','label_relative_path','labels','size_xyz','spacing_xyz_mm','foreground_fraction','geometry_matches'])
    writer.writeheader()
    for row in annotated + blind:
        writer.writerow({key: row.get(key, None) if key not in ['labels','size_xyz','spacing_xyz_mm'] else
                         (json.dumps(row.get('labels')) if key=='labels' else json.dumps(row['image_header'][key])) for key in writer.fieldnames})
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
