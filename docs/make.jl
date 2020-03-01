push!(LOAD_PATH, "../src/", "../test/src/")

DOCUMENTER_DEBUG=true

using Documenter, PartialSvdStoch


makedocs(
    format = Documenter.HTML(prettyurls = false),
    sitename = "PartialSvdStoch",
    pages = Any[
        "Introduction" => "INTRO.md",
        "PartialSvdStoch.jl " => "index.md",
        "Tests" => "Test.md"
    ]
)

