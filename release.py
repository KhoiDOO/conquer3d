#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse

def get_version():
    """Parse version string from pyproject.toml without external dependencies."""
    with open('pyproject.toml', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('version ='):
                return line.split('=')[1].strip().strip('"').strip("'")
    raise ValueError("Version not found in pyproject.toml")

def main():
    parser = argparse.ArgumentParser(description="Create a new release.")
    parser.add_argument("-n", "--notes", help="Release notes to include", default=None)
    args = parser.parse_args()

    try:
        version = get_version()
        tag = f"v{version}"
        print(f"Starting release process for {tag}...")

        # 1. Create git tag
        print(f"Creating git tag: {tag}")
        subprocess.run(['git', 'tag', '-a', tag, '-m', f"Release {tag}"], check=True)

        # 2. Push the tag to GitHub
        print("Pushing tag to origin...")
        subprocess.run(['git', 'push', 'origin', tag], check=True)

        # 3. Perform a GitHub release
        print(f"Creating GitHub release for {tag}...")
        try:
            # Requires GitHub CLI (gh)
            cmd = ['gh', 'release', 'create', tag, '--title', f"Release {tag}"]
            if args.notes:
                cmd.extend(['--notes', args.notes])
            else:
                cmd.append('--generate-notes')
                
            subprocess.run(cmd, check=True)
            print(f"Successfully released {tag} on GitHub!")
        except FileNotFoundError:
            print("Warning: 'gh' (GitHub CLI) is not installed. The tag was pushed, but the GitHub release was not created.")
            print("Please install 'gh' or manually create the release on GitHub.")

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
