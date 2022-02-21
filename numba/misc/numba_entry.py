import sys
import argparse
import os
import subprocess
import json
import importlib

from .numba_sysinfo import display_sysinfo, get_sysinfo


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotate', help='Annotate source',
                        action='store_true')
    parser.add_argument('--dump-llvm', action="store_true",
                        help='Print generated llvm assembly')
    parser.add_argument('--dump-optimized', action='store_true',
                        help='Dump the optimized llvm assembly')
    parser.add_argument('--dump-assembly', action='store_true',
                        help='Dump the LLVM generated assembly')
    parser.add_argument('--annotate-html', nargs=1,
                        help='Output source annotation as html')
    parser.add_argument('-s', '--sysinfo', action="store_true",
                        help='Output system information for bug reporting')
    parser.add_argument('--sys-json', nargs=1,
                        help='Saves the system info dict as a json file')
    parser.add_argument('filename', nargs='?', help='Python source filename')
    ### AOT
    sub_parsers = parser.add_subparsers(help='Numba AOT help', dest='kind')
    parser_llvm = sub_parsers.add_parser('emit-obj', help='emit-obj')
    parser_merge = sub_parsers.add_parser('merge', help='merge object files (*.o)')

    parser_llvm.add_argument('-f',
                             '--function',
                             action='store',
                             type=str,
                             required=True,
                             help='The function to be exported')

    parser_llvm.add_argument('-n',
                             '--name',
                             action='store',
                             type=str,
                             required=True,
                             help='Name of the exported function')

    parser_llvm.add_argument('-s',
                             '--signature',
                             action='store',
                             type=str,
                             required=True,
                             help='Signature of the exported function')

    parser_llvm.add_argument('-o',
                             action='store',
                             type=str,
                             required=True,
                             help='Name of the output file')

    parser_merge.add_argument(action='store',
                             type=str,
                             nargs='+',
                             dest='files',
                             help='list of llvm IR files to be merged')

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.sysinfo:
        print("System info:")
        display_sysinfo()
        sys.exit(0)

    if args.sys_json:
        info = get_sysinfo()
        info.update({'Start': info['Start'].isoformat()})
        info.update({'Start UTC': info['Start UTC'].isoformat()})
        with open(args.sys_json[0], 'w') as f:
            json.dump(info, f, indent=4)
        sys.exit(0)

    os.environ['NUMBA_DUMP_ANNOTATION'] = str(int(args.annotate))
    if args.annotate_html is not None:
        try:
            from jinja2 import Template
        except ImportError:
            raise ImportError("Please install the 'jinja2' package")
        os.environ['NUMBA_DUMP_HTML'] = str(args.annotate_html[0])
    os.environ['NUMBA_DUMP_LLVM'] = str(int(args.dump_llvm))
    os.environ['NUMBA_DUMP_OPTIMIZED'] = str(int(args.dump_optimized))
    os.environ['NUMBA_DUMP_ASSEMBLY'] = str(int(args.dump_assembly))

    from numba.pycc import CC
    cc = CC('my_module', output_dir=os.getcwd())

    if args.filename:
        if args.kind is not None:
            # import module programatically
            # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
            mod_name = args.filename.strip('.py')
            mod_path = os.path.abspath(args.filename)
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

        if args.kind == 'emit-obj':
            fn_name = args.function
            exported_name = args.name
            sig = args.signature
            try:
                fn = getattr(module, fn_name)
            except AttributeError:
                raise ImportError(f'function {fn_name} not found in {module.__name__}')
            cc.export(exported_name, sig)(fn)
            cc.emit_object_file(args.o)
        elif args.kind == 'merge':
            cc.merge_object_files(args.files)
        else:
            cmd = [sys.executable, args.filename]
            subprocess.call(cmd)
    else:
        print("numba: error: the following arguments are required: filename")
        sys.exit(1)
