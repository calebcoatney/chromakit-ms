import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from logic.c_folder import CFolder

with tempfile.TemporaryDirectory() as tmpdir:
    src_files = []
    for name in ['sample.rsd', 'sample_0.5_secs.dbc.lsc', 'sample.HDR']:
        p = os.path.join(tmpdir, name)
        open(p, 'w').close()
        src_files.append(p)

    folder = CFolder.create_multi(src_files, signal_type='gcxgc', sample_id='TEST')

    assert os.path.isdir(os.path.join(folder.path, 'data'))
    assert os.path.isfile(os.path.join(folder.path, 'manifest.json'))
    assert folder.get_manifest()['source_format'] == 'sepsolve'
    assert folder.get_manifest()['signal_type'] == 'gcxgc'
    data_files = os.listdir(os.path.join(folder.path, 'data'))
    assert 'sample.rsd' in data_files
    assert 'sample_0.5_secs.dbc.lsc' in data_files
    assert 'sample.HDR' in data_files
    print('✓ create_multi structure correct')

    try:
        folder.extract()
        print('✗ extract() should raise NotImplementedError')
        sys.exit(1)
    except NotImplementedError:
        print('✓ extract() raises NotImplementedError for sepsolve')
