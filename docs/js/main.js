
let allSVGElements = document.querySelectorAll("svg");
for (const svgElem of allSVGElements) {
    svgElem.setAttribute("width", "100%");
}

let allTextElements = document.querySelectorAll("svg text");
for (const textElem of allTextElements) {

    if (textElem.textContent.startsWith("&")) {

        let ellipseElem = textElem.parentElement.getElementsByTagName("ellipse")[0];
        let x, y, spanClass;
        if (ellipseElem !== undefined) {
            x = ellipseElem.getAttribute("cx");
            y = ellipseElem.getAttribute("cy");
            spanClass = "foreign-span-center";
        } else {
            x = textElem.getAttribute("x");
            y = textElem.getAttribute("y");
            spanClass = "foreign-span-left";
        }

        let foreignElem = document.createElementNS("http://www.w3.org/2000/svg", 'foreignObject');
        foreignElem.setAttribute("x", x - 18);
        foreignElem.setAttribute("y", y - 18);
        foreignElem.setAttribute("width", 36);
        foreignElem.setAttribute("height", 36);

        let divElem = document.createElement("div");
        divElem.classList.add("foreign-div")

        let spanElem = document.createElement("span");
        spanElem.classList.add(spanClass);

        let mathjaxElem = MathJax.tex2chtml(textElem.textContent.slice(1), { em: 10, ex: 6, display: false });

        spanElem.append(mathjaxElem);
        divElem.append(spanElem);
        foreignElem.append(divElem);
        textElem.parentElement.append(foreignElem);

        textElem.parentElement.removeChild(textElem);
    }
}
