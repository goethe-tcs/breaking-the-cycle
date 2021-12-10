# Directed Feedback Vertex Set

To contribute to this project, please [create your own fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html).

## Prepare your system

After cloning your fork to your computer, make sure to also pull the `data` repository.
To do so, run `git submodule update --init`.

Our code basis uses recent Rust features that are only available in the so-called [nightly channel](https://rust-lang.github.io/rustup/concepts/channels.html).
To install it, use `rustup toolchain install nightly`; then you can activate it using `rustup default nightly`.
Finally, we need the package tarpaulin to carry out coverage tests; to install it run `cargo install cargo-tarpaulin`.
Please be aware that these commands affect your whole Rust installation (they are not limited to this project)!

## Contributing

In your fork, you are free to modify as you wish, but please develop features in their own branches and use meaningful commit messages.
A typical contribution should only consist of a handful of commits; though, if there's good reason, more are okay.
But, please rebase/squash chains of "Test 1", "Test 2", "Try again" commits ...

Each time you push changes to the GitLab server, your code is compiled and executed in several configurations.
We highly recommend running `./checks.sh` before pushing to detect the most common issues even before pushing.
This will create a file `tarpaulin-report.html` that includes a coverage analysis.
Make sure to include enough unit-tests to cover all your code (exceptions may be permissible but need to be justified in the merge request).

Once a feature is complete (and all CI tests pass!), you can create a [merge request](https://docs.gitlab.com/ee/user/project/merge_requests/getting_started.html) to incorporate the feature into the main repository after review.
