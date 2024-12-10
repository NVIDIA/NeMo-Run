# CONTRIBUTING GUIDELINES

We are glad you are contributing to NeMo Run! Before you make a PR, be sure to read over this guide in detail.
This checklist ensures that NeMo Run stays easy-to-use by both users and developers.
Not all steps are necessary for some contributions, so read the linked sections for more information about each item.

- [CONTRIBUTING GUIDELINES](#contributing-guidelines)
  - [General principles](#general-principles)
  - [Environment Setup](#environment-setup)
  - [Code Structure](#code-structure)
  - [Examples and Documentation](#examples-and-documentation)
  - [Python style](#python-style)
  - [Unit tests](#unit-tests)
  - [Pull Requests (PR) Guidelines](#pull-requests-pr-guidelines)
  - [Sign Your Work](#sign-your-work)
  - [Full text of the DCO:](#full-text-of-the-dco)
  - [Whom should you ask for review:](#whom-should-you-ask-for-review)

## General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easier to read.
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo Run supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Environment Setup
We use [uv](https://docs.astral.sh/uv/) to develop NeMo Run. The following steps should get you started with the dev environment:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone NeMo-Run
3. Sanity check with `uv sync --extra skypilot && uv run -- pytest test/` (This will create a venv and run all unit tests)

If all tests passed, then you should be good to get started with the development of NeMo-Run.

## Code Structure
The repository is home to flexible Python modules, sample scripts, tests, and more.
Here is a brief overview of where everything lives:
- [docker](docker/) - Dockerfiles to build NeMo with NeMo Run.
- [docs](docs/) - Walkthroughs and guides the library.
- [examples](examples/) - Examples for how users may want to use NeMo Run.
- [src](src/) -
    - [nemo_run](src/nemo_run/) - The source code for NeMo Run.
- [test](test/) - Unit tests.

## Examples and Documentation
Examples provide an easy way for users to see how the NeMo Run works in action.
They should be incredibly lightweight and rely mostly on `nemo_run` for their functionality
Most should be designed for a user to get up and running on their local machines, but examples on remote clusters are welcomed if it makes sense.
Python scripts should be the primary way to run your example.

The documentation should complement each example by going through the motivation behind why a user would use each a particular API in NeMo Run.
It should include both an explanation of the API, and how it's used in its corresponding example.
The documentation should also cover potential pitfalls and caveats.
This existing examples and documentation should serve as a good reference to what is expected.

## Python style
We use [``ruff``](https://docs.astral.sh/ruff/) for linting and formatting. To lint and format your code, you can run `uv run --group lint -- ruff check` and `uv run --group lint -- ruff format` respectively.

## Unit tests
Unit tests should be simple and fast.
Developers should be able to run them frequently while developing without any slowdown.

## Pull Requests (PR) Guidelines

**Send your PRs to the `main` or `dev` branch**

1) Make sure your PR does one thing. Have a clear answer to "What does this PR do?".
2) Read General Principles and style guide above
3) Make sure you sign off your commits. E.g. use ``git commit --signoff`` when committing.
4) Make sure all unit tests finish successfully before sending PR
5) Send your PR and request a review

The `dev` branch is for active development and may be unstable. Unit tests are expected to pass before merging into `dev` or `main`.
Every release `dev` and `main` will sync to be the same.

## Sign Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` option when committing your changes:
  ```bash
  $ git commit --signoff -m "Add cool feature."
  ```

## Full text of the DCO:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Whom should you ask for review:

Hemil Desai (@hemildesai) or Marc Romeijn (@marcromeyn)

They may ask for other reviewers depending on the scope of the change. Your pull requests must pass all checks and peer-review before they can be merged.

Thank you for contributing to NeMo Run!
