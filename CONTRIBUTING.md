# Contributing to EXO

Thank you for your interest in contributing to EXO!

## Getting Started

To run EXO from source:
```bash
git clone https://github.com/exo-explore/exo.git
cd exo/dashboard
npm install && npm run build
uv run exo
```

## Development

EXO is built with a mix of Rust, Python, and TypeScript (Svelte for the dashboard), and the codebase is actively evolving. Before starting work:

- Pull the latest source to ensure you're working with the most recent code
- Keep your changes focused - implement one feature or fix per pull request
- Avoid combining unrelated changes, even if they seem small

This makes reviews faster and helps us maintain code quality as the project evolves.

## Code Style

Write pure functions where possible. When adding new code, prefer Rust unless there's a good reason otherwise. Leverage the type systems available to you - Rust's type system, Python type hints, and TypeScript types. Comments should explain why you're doing something, not what the code does - especially for non-obvious decisions.

Run `nix fmt` to auto-format your code before submitting.

## Testing

EXO relies heavily on manual testing at this point in the project, but this is evolving. Before submitting a change, test both before and after to demonstrate how your change improves behavior. Do the best you can with the hardware you have available - if you need help testing, ask and we'll do our best to assist. Add automated tests where possible - we're actively working to substantially improve our automated testing story.

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request and follow the PR template

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear description of the problem or feature
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (macOS version, hardware, etc.)

## Questions?

Join our community:
- [X](https://x.com/exolabs)
