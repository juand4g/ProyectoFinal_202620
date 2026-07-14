# astro-thesis

A personal thesis template that mimics Astronomy&Astrophysics citation design, and have reasonable defaults I feel confortable working with.
I used this on my [master's thesis](https://repositorio.uniandes.edu.co/entities/publication/713b289b-ac71-4862-905a-0860f50a5b7b).


The citation file `aa_url.bst` was taken from [yangcht's repository](https://github.com/yangcht/AA-bibstyle-with-hyperlink), and it does its job well, but the `inproceedings` function was modified to prevent errors with links.

The template contains dummy text with a reasonable structure, and a sample Figure, Table and code Listing.

My customizations can be summarized as:
* `\intent` command to provide margin comments, that can be deactivated by `\visibleintentfalse`
* Figure and table captions at 90% of the textwidth
* Line 1300 of the `aa_url.bst` file, deleteing the `adsurl make.href.hyperref` call to prevent errors with my bib file.

## Customization

If there is some problem with bibliography, for example 'command \avvso not found', you can add it to the end of the `.sty` file
```latex
\newcommand*{\aavso}{AVVSO}
```


## Licensing

Sample figures include data from the [OGLE project](https://ogle.astrouw.edu.pl/main/OGLEIV/mosaic.html). Otherwise, you are free to use this however you want, no need for attribution.

