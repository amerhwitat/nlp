"""Single-command builder for the Chimera II OS + BizX + BizXtreme ISO workspace."""
from __future__ import annotations
import argparse
from pathlib import Path
from iso_tool.workspace_builder import DEFAULT_REPOSITORIES, build_workspace, load_profiles


def main(argv=None):
    ap = argparse.ArgumentParser(description='Build and stage related Chimera II repositories into one bootable ISO/IMG workspace')
    ap.add_argument('--output', required=True, help='Final workspace/output directory')
    ap.add_argument('--compiler', choices=('auto', 'gnu', 'msvc'), default='auto')
    ap.add_argument('--profile', type=Path, help='JSON repository profile; defaults to engine/repository-profiles.json')
    ap.add_argument('--repos', nargs='*', metavar='ID=URL', help='Override/add repository sources')
    args = ap.parse_args(argv)
    profiles = load_profiles(args.profile) if args.profile else load_profiles(Path(__file__).resolve().parents[1] / 'engine' / 'repository-profiles.json')
    repos = {r['id']: r['url'] for r in profiles.get('repositories', [])} or dict(DEFAULT_REPOSITORIES)
    for item in args.repos or []:
        repo_id, url = item.split('=', 1)
        repos[repo_id] = url
    result = build_workspace(repos, Path(args.output), args.compiler)
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
