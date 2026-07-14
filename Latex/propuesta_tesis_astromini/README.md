# astro-mini

A personal miscellaneous document template that mimics Astronomy&Astrophysics citation design.

It is made for proposals, reports, homeworks and other short-styled works.


The citation file `aa_url.bst` was taken from [yangcht's repository](https://github.com/yangcht/AA-bibstyle-with-hyperlink), and it does its job well, but the `inproceedings` function is modified.

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

At the beginning of the `.sty` file is the language configuration. There is commented a way to put everything in spanish.


## Licensing

Sample figures include data from the [OGLE project](https://ogle.astrouw.edu.pl/main/OGLEIV/mosaic.html).

 
