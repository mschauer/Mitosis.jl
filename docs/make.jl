using Mitosis
using Documenter

makedocs(;
    modules=[Mitosis],
    authors="mschauer <moritzschauer@web.de> and contributors",
    repo="https://github.com/mschauer/Mitosis.jl/blob/{commit}{path}#L{line}",
    sitename="Mitosis.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mschauer.github.io/Mitosis.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mschauer/Mitosis.jl",
)
