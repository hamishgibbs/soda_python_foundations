#!/bin/bash

echo "Rendering Quarto project..."
quarto render

HTML_DIR="_site"

DECKTAPE_CMD="decktape"

if ! command -v $DECKTAPE_CMD &> /dev/null
then
    echo "DeckTape could not be found, please install it using 'npm install -g decktape'"
    exit
fi

for file in "$HTML_DIR"/slides/slides_*.html; do
    if [ "$file" = "$HTML_DIR/slides/slides_*.html" ]; then
        echo "No slides found in $HTML_DIR"
        break
    fi

    pdf="${file%.html}.pdf"

    echo "Converting $file to $pdf..."
    $DECKTAPE_CMD reveal "$file" "$pdf"
done

echo "Conversion complete."
