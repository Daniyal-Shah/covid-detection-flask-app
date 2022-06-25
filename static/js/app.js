function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#imageResult").attr("src", e.target.result);
    };

    reader.readAsDataURL(input.files[0]);
  }
}

$(function () {
  $("#upload").on("change", function () {
    let btn = document.getElementById("btnSubmit");
    btn.classList.remove("disabled");
    readURL(input);
  });
});

var input = document.getElementById("upload");
var infoArea = document.getElementById("upload-label");

input.addEventListener("change", showFileName);

function showFileName(event) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = "File name: " + fileName;
}

$(function () {
  $("#btnReset").on("click", function () {
    $("#imageResult").attr("src", "../static/img_icon.png");

    let btn = document.getElementById("btnSubmit");
    btn.classList.add("disabled");
    infoArea.textContent = "Choose file ";
  });
});
