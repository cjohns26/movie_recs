"""Package entrypoint."""

from movie_recs import __version__


def health() -> str:
    """Return a liveness string pinned to the installed package version.

    Used by the CLI entrypoint and, later, the `/health` API route.
    """
    return f"movie-recs v{__version__} OK"


def main() -> None:
    print(health())


if __name__ == "__main__":
    main()
