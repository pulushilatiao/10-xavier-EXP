#!/usr/bin/env python3
"""TensorRT 8.5 inspector -> honest CSV/Markdown/SVG PPT report (stdlib only).
Usage: python3 inspect_trt_precision_v2.py 1012_int8.engine --out int8_report
       python3 inspect_trt_precision_v2.py detailed_layers.json --out int8_report
"""
import argparse, csv, json, re, sys, textwrap, gc
from pathlib import Path
from collections import Counter

DTYPES = ('INT8','FP16','FP32','INT32','BOOL','FP8','BF16')

def dtype_from_text(value):
    s=str(value).lower()
    tests=[('INT8',r'\bint\s*8\b|\bkint8\b|\bint8\b'),
           ('FP16',r'\bhalf\b|\bfp16\b|\bfloat16\b|\bkhalf\b'),
           ('FP32',r'\bfloat\b|\bfp32\b|\bfloat32\b|\bkfloat\b'),
           ('INT32',r'\bint32\b|\bkint32\b'),
           ('BOOL',r'\bbool\b'),('FP8',r'\bfp8\b'),('BF16',r'\bbfloat16\b|\bbf16\b')]
    for name, pattern in tests:
        if re.search(pattern,s): return name
    return 'UNKNOWN'

def port_types(obj):
    if not isinstance(obj,list): return 'UNKNOWN'
    result=[]
    for p in obj:
        if not isinstance(p,dict): continue
        # Dtype can be nested in a tensor metadata dictionary.
        val=p.get('Format/Datatype',p.get('DataType',p.get('Datatype',p.get('dtype',p.get('Type','')))))
        x=dtype_from_text(val)
        if x not in result: result.append(x)
    return '+'.join(result) if result else 'UNKNOWN'

def parse_layer_json(data):
    if isinstance(data,list): return data
    if isinstance(data,dict):
        for k in ('Layers','layers','LayerInformation','layerInformation'):
            if isinstance(data.get(k),list): return data[k]
        # One record only
        if 'Name' in data or 'LayerType' in data: return [data]
    return []

def inspect_engine(path):
    try: import tensorrt as trt
    except ImportError as e: raise RuntimeError('TensorRTが読み込めません。Jetson上で実行するか、JSONを入力してください。') from e
    print('TensorRT:',trt.__version__)
    logger=trt.Logger(trt.Logger.WARNING)
    runtime=engine=inspector=None
    try:
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None: raise RuntimeError('エンジンを読み込めません。TensorRTとGPUの互換性を確認してください。')
        count=engine.num_layers
        inspector=engine.create_engine_inspector()
        try: raw=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        except Exception: raw=''
        try: data=json.loads(raw)
        except (ValueError,TypeError): data=None
        records=parse_layer_json(data)
        if not records or len(records)!=count:
            records=[]
            for i in range(count):
                s=inspector.get_layer_information(i,trt.LayerInformationFormat.JSON)
                try: rec=json.loads(s)
                except (ValueError, TypeError): rec={'Name':s}
                records.append(rec)
        return records, {'tensorrt_version':trt.__version__,'engine_layers':count}
    finally:
        # Release children before engine and runtime. In TRT 8.5 order matters.
        inspector=None
        gc.collect()
        engine=None
        gc.collect()
        runtime=None
        gc.collect()

def simplify(records):
    rows=[]
    for i,rec in enumerate(records):
        if isinstance(rec,str): rec={'Name':rec}
        if not isinstance(rec,dict): rec={'Name':str(rec)}
        ins=port_types(rec.get('Inputs'))
        outs=port_types(rec.get('Outputs'))
        name=str(rec.get('Name',f'layer_{i}'))
        layer_type=str(rec.get('LayerType',rec.get('ParameterType','UNKNOWN')))
        info=str(rec.get('Metadata',''))
        # Only label known from actual input/output formats. Never infer layer compute precision.
        if ins=='UNKNOWN' and outs=='UNKNOWN': category='UNKNOWN'
        elif ins==outs and ins in DTYPES: category=f'{ins} I/O'
        elif ins=='UNKNOWN' or outs=='UNKNOWN': category='PARTIAL'
        else: category='MIXED I/O'
        rows.append({'index':i,'layer_name':name,'layer_type':layer_type,
                     'input_dtypes':ins,'output_dtypes':outs,'io_class':category,
                     'compute_precision':'NOT_PROVEN','notes':info})
    return rows

def svg_escape(t):
    return str(t).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')

def make_svg(rows,output):
    count=Counter(x['io_class'] for x in rows)
    order=['INT8 I/O','FP16 I/O','FP32 I/O','MIXED I/O','PARTIAL','UNKNOWN']
    colors={'INT8 I/O':'#2875B9','FP16 I/O':'#208A77','FP32 I/O':'#B0822C','MIXED I/O':'#805DB3','PARTIAL':'#8A94A3','UNKNOWN':'#A0A5AD'}
    W=1080; H=540
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<text x="54" y="70" font-family="DejaVu Sans, sans-serif" font-size="30" font-weight="bold" fill="#202B37">TensorRT engine: layer I/O datatypes</text>',
           f'<text x="54" y="106" font-size="20" font-family="DejaVu Sans, sans-serif" fill="#53616F">{len(rows)} optimized layers  |  verified from engine inspector</text>']
    maxv=max([count.get(x,0) for x in order]+[1])
    for i,key in enumerate(order):
        y=152+i*51; v=count.get(key,0)
        lines += [f'<text x="54" y="{y+21}" font-size="20" font-family="DejaVu Sans, sans-serif" fill="#253340">{svg_escape(key)}</text>',
                  f'<rect x="240" y="{y}" width="640" height="28" rx="4" fill="#EFF2F5"/>',
                  f'<rect x="240" y="{y}" width="{640*v/maxv:.1f}" height="28" rx="4" fill="{colors[key]}"/>',
                  f'<text x="904" y="{y+22}" font-size="20" font-family="DejaVu Sans, sans-serif" fill="#253340">{v}</text>']
    lines.append('<text x="54" y="494" font-family="DejaVu Sans, sans-serif" font-size="17" fill="#6A7682">I/O datatype is NOT proof of internal computation precision.</text>')
    lines.append('</svg>')
    output.write_text('\n'.join(lines),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description='TensorRT precision inspection + PPT summary')
    ap.add_argument('source',type=Path,help='.engine or JSON from trtexec/EngineInspector')
    ap.add_argument('--out',default='int8_report',help='output directory')
    a=ap.parse_args(); src=a.source.expanduser().resolve()
    if not src.is_file(): ap.error(f'File not found: {src}')
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if src.suffix.lower()=='.engine': records,meta=inspect_engine(src)
    else:
        raw=json.loads(src.read_text(encoding='utf-8'))
        records=parse_layer_json(raw)
        if not records: raise RuntimeError('JSONにLayersが見つかりません。trtexec --exportLayerInfo の結果を確認してください。')
        meta={'source_type':'json','engine_layers':len(records)}
    rows=simplify(records)
    with (out/'layers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['index']);writer.writeheader();writer.writerows(rows)
    with (out/'raw.json').open('w',encoding='utf-8') as f: json.dump(records,f,indent=2,ensure_ascii=False)
    count=Counter(x['io_class'] for x in rows)
    complete=sum(1 for x in rows if x['input_dtypes']!='UNKNOWN' and x['output_dtypes']!='UNKNOWN')
    summary={'engine_path':str(src),'total_layers':len(rows),'known_both_io_datatypes':complete,
             'classification_counts':dict(count),'metadata':meta,
             'critical_note':'Input/output tensor datatypes do not prove internal compute precision.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    make_svg(rows,out/'ppt_chart.svg')
    lines=['# TensorRT 混合精度エンジン解析結果','',f'- Engine: `{src.name}`',
           f'- 最適化後レイヤー数: **{len(rows)}**',f'- 入力・出力データ型を両方確認できたレイヤー: **{complete}/{len(rows)}**','','## 集計（入出力形式別）','',
           '| 種類 | レイヤー数 |','|---|---:|']
    for k,v in sorted(count.items()):lines.append(f'| {k} | {v} |')
    lines += ['','## PPT用の正確な説明','',
              '本集計はTensorRTの最適化後エンジンの入出力データ型に基づく。内部の積和演算精度や全演算量のINT8比率を直接示すものではない。',
              'Conv・SiLUなどは融合される場合があり、元のYOLOv8モジュールと1対1には対応しない。','']
    if complete==0:
        lines += ['## 未解決：精度情報なし','',
                  'エンジンは解析できたが、データ型の詳細が保存されていない可能性がある。**INT8/FP16割合を推測してPPTに記載しないこと。**',
                  '元のONNXとキャリブレーションキャッシュを使用し、--profilingVerbosity=detailedを付けて再ビルドする。']
    lines += ['','## スライドタイトル案','', '「TensorRT混合精度エンジンのレイヤー構成とデータ型解析」']
    (out/'ppt_notes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Layers:',len(rows),'| known I/O datatypes:',complete)
    print('Counts:',dict(count))
    print('Saved to:',out)
    if complete==0: print('WARNING: Detailed dtype metadata unavailable; rebuild with --profilingVerbosity=detailed.')

if __name__=='__main__':
    try: main()
    except Exception as e:
        print('ERROR:',str(e),file=sys.stderr);sys.exit(1)
