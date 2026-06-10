#!/usr/bin/env python3
"""
CI/CD integration script for automatic documentation updates.

This script is designed to be run in CI/CD pipelines to automatically update
documentation when code changes are detected.

Usage:
    uv run python scripts/ci_docs_update.py --codebase /path/to/repo --output /path/to/docs
    uv run python scripts/ci_docs_update.py --help
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logging():
    """Setup logging for CI/CD environment."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def detect_ci_environment() -> Dict[str, Any]:
    """Detect the CI/CD environment and extract relevant information."""
    ci_info: Dict[str, Any] = {
        "provider": None,
        "branch": None,
        "commit": None,
        "base_commit": None,
        "pr_number": None,
        "is_pull_request": False,
    }

    # GitHub Actions
    if os.getenv("GITHUB_ACTIONS"):
        ci_info["provider"] = "github"
        ci_info["branch"] = os.getenv("GITHUB_REF_NAME")
        ci_info["commit"] = os.getenv("GITHUB_SHA")
        ci_info["base_commit"] = os.getenv("GITHUB_BASE_REF")

        if os.getenv("GITHUB_EVENT_NAME") == "pull_request":
            ci_info["is_pull_request"] = True
            ci_info["pr_number"] = os.getenv("GITHUB_EVENT_NUMBER")

    # GitLab CI
    elif os.getenv("GITLAB_CI"):
        ci_info["provider"] = "gitlab"
        ci_info["branch"] = os.getenv("CI_COMMIT_REF_NAME")
        ci_info["commit"] = os.getenv("CI_COMMIT_SHA")
        ci_info["base_commit"] = os.getenv("CI_MERGE_REQUEST_TARGET_BRANCH_SHA")

        if os.getenv("CI_MERGE_REQUEST_ID"):
            ci_info["is_pull_request"] = True
            ci_info["pr_number"] = os.getenv("CI_MERGE_REQUEST_ID")

    # Jenkins
    elif os.getenv("JENKINS_URL"):
        ci_info["provider"] = "jenkins"
        ci_info["branch"] = os.getenv("GIT_BRANCH")
        ci_info["commit"] = os.getenv("GIT_COMMIT")
        ci_info["base_commit"] = os.getenv("GIT_PREVIOUS_COMMIT")

    # Azure DevOps
    elif os.getenv("TF_BUILD"):
        ci_info["provider"] = "azure"
        ci_info["branch"] = os.getenv("BUILD_SOURCEBRANCH")
        ci_info["commit"] = os.getenv("BUILD_SOURCEVERSION")

        if os.getenv("SYSTEM_PULLREQUEST_PULLREQUESTID"):
            ci_info["is_pull_request"] = True
            ci_info["pr_number"] = os.getenv("SYSTEM_PULLREQUEST_PULLREQUESTID")

    # CircleCI
    elif os.getenv("CIRCLECI"):
        ci_info["provider"] = "circleci"
        ci_info["branch"] = os.getenv("CIRCLE_BRANCH")
        ci_info["commit"] = os.getenv("CIRCLE_SHA1")

        if os.getenv("CIRCLE_PULL_REQUEST"):
            ci_info["is_pull_request"] = True
            # Extract PR number from URL
            pr_url = os.getenv("CIRCLE_PULL_REQUEST", "")
            if "/pull/" in pr_url:
                ci_info["pr_number"] = pr_url.split("/pull/")[-1]

    return ci_info


def get_base_commit_for_comparison(codebase_path: str, ci_info: Dict[str, Any]) -> str:
    """Determine the appropriate base commit for change detection."""
    logger = setup_logging()

    # Try CI-specific base commit first
    if ci_info["base_commit"]:
        logger.info(f"Using CI-provided base commit: {ci_info['base_commit']}")
        return ci_info["base_commit"]

    # For pull requests, try to find the merge base
    if ci_info["is_pull_request"] and ci_info["provider"] == "github":
        try:
            # Try to get the merge base from the PR
            result = subprocess.run(
                ["git", "merge-base", "HEAD", "origin/main"],
                cwd=codebase_path,
                capture_output=True,
                text=True,
                check=True,
            )
            base_commit = result.stdout.strip()
            logger.info(f"Using merge base: {base_commit}")
            return base_commit
        except subprocess.CalledProcessError:
            logger.warning("Could not determine merge base, falling back to HEAD~1")

    # Default fallback
    logger.info("Using default base commit: HEAD~1")
    return "HEAD~1"


def run_docstra_command(cmd: List[str], cwd: str) -> bool:
    """Run a docstra command and return success status."""
    logger = setup_logging()

    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        )

        # Log output
        if result.stdout:
            logger.info(f"Command output:\n{result.stdout}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(
            "Could not find the configured Docstra entrypoint. Run this script with "
            "'uv run' from the repo root."
        )
        return False


def check_documentation_status(
    codebase_path: str, config_path: Optional[str] = None
) -> Dict:
    """Check the current documentation status."""
    logger = setup_logging()

    cmd = [sys.executable, "-m", "docstra", "status", codebase_path, "--detailed"]
    if config_path:
        cmd.extend(["--config", config_path])

    try:
        result = subprocess.run(
            cmd, cwd=codebase_path, capture_output=True, text=True, check=True
        )

        logger.info("Documentation status checked successfully")
        return {"success": True, "output": result.stdout}

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check documentation status: {e}")
        return {"success": False, "error": str(e)}


def update_documentation(
    codebase_path: str,
    output_dir: str,
    base_ref: str,
    config_path: Optional[str] = None,
    force_files: Optional[List[str]] = None,
) -> bool:
    """Update documentation incrementally."""
    cmd = [
        sys.executable,
        "-m",
        "docstra",
        "update",
        codebase_path,
        "--base",
        base_ref,
        "--output",
        output_dir,
    ]

    if config_path:
        cmd.extend(["--config", config_path])

    if force_files:
        for file_path in force_files:
            cmd.extend(["--force", file_path])

    return run_docstra_command(cmd, codebase_path)


def generate_full_documentation(
    codebase_path: str, output_dir: str, config_path: Optional[str] = None
) -> bool:
    """Generate full documentation (fallback when incremental fails)."""
    cmd = [
        sys.executable,
        "-m",
        "docstra",
        "generate",
        codebase_path,
        "--output",
        output_dir,
        "--no-wizard",
    ]

    if config_path:
        cmd.extend(["--config", config_path])

    return run_docstra_command(cmd, codebase_path)


def create_ci_summary(
    codebase_path: str,
    output_dir: str,
    success: bool,
    ci_info: Dict[str, Any],
    execution_time: float,
) -> None:
    """Create a summary file for CI/CD reporting."""
    summary = {
        "timestamp": int(time.time()),
        "success": success,
        "execution_time_seconds": execution_time,
        "ci_environment": ci_info,
        "codebase_path": str(Path(codebase_path).resolve()),
        "output_path": str(Path(output_dir).resolve()),
    }

    summary_file = Path(output_dir) / "ci_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"CI summary written to: {summary_file}")


def main():
    """Main entry point for CI/CD documentation updates."""
    parser = argparse.ArgumentParser(
        description="CI/CD integration script for automatic documentation updates"
    )

    parser.add_argument(
        "--codebase", required=True, help="Path to the codebase directory"
    )

    parser.add_argument(
        "--output", required=True, help="Output directory for documentation"
    )

    parser.add_argument("--config", help="Path to docstra configuration file")

    parser.add_argument(
        "--base-ref",
        help="Base git reference for change detection (auto-detected if not provided)",
    )

    parser.add_argument(
        "--force-files", nargs="*", help="Force regeneration for specific files"
    )

    parser.add_argument(
        "--full-generation",
        action="store_true",
        help="Force full documentation generation instead of incremental",
    )

    parser.add_argument(
        "--fail-on-no-changes",
        action="store_true",
        help="Exit with error code if no changes are detected",
    )

    parser.add_argument(
        "--skip-status-check",
        action="store_true",
        help="Skip initial documentation status check",
    )

    args = parser.parse_args()

    logger = setup_logging()
    start_time = time.time()

    # Resolve paths
    codebase_path = Path(args.codebase).resolve()
    output_dir = Path(args.output).resolve()

    logger.info("Starting CI/CD documentation update")
    logger.info(f"Codebase: {codebase_path}")
    logger.info(f"Output: {output_dir}")

    # Detect CI environment
    ci_info = detect_ci_environment()
    logger.info(f"CI Environment: {ci_info['provider'] or 'unknown'}")

    # Check if codebase path exists
    if not codebase_path.exists():
        logger.error(f"Codebase path does not exist: {codebase_path}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    success = False

    try:
        # Check documentation status first (unless skipped)
        if not args.skip_status_check:
            logger.info("Checking documentation status...")
            status_result = check_documentation_status(str(codebase_path), args.config)
            if not status_result["success"]:
                logger.warning(
                    "Could not check documentation status, proceeding anyway"
                )

        # Determine base reference for change detection
        if args.base_ref:
            base_ref = args.base_ref
        else:
            base_ref = get_base_commit_for_comparison(str(codebase_path), ci_info)

        logger.info(f"Using base reference: {base_ref}")

        # Decide on generation strategy
        if args.full_generation:
            logger.info("Running full documentation generation (forced)")
            success = generate_full_documentation(
                str(codebase_path), str(output_dir), args.config
            )
        else:
            logger.info("Running incremental documentation update")
            success = update_documentation(
                str(codebase_path),
                str(output_dir),
                base_ref,
                args.config,
                args.force_files,
            )

            # Fallback to full generation if incremental fails
            if not success:
                logger.warning(
                    "Incremental update failed, falling back to full generation"
                )
                success = generate_full_documentation(
                    str(codebase_path), str(output_dir), args.config
                )

        execution_time = time.time() - start_time

        # Create CI summary
        create_ci_summary(
            str(codebase_path), str(output_dir), success, ci_info, execution_time
        )

        if success:
            logger.info(
                f"Documentation update completed successfully in {execution_time:.2f} seconds"
            )
            print(
                f"::notice::Documentation updated successfully in {execution_time:.2f}s"
            )
        else:
            logger.error("Documentation update failed")
            print("::error::Documentation update failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error during documentation update: {e}")
        print(f"::error::Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
