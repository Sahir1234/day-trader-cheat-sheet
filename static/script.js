$(document).ready(function() {
  $(".submission").on("submit", function() {
    $(".submission").toggle();
    $("#stockPrompt").text('LOADING DATA...')
    $("#modelPrompt").text('TRAINING MODELS...')
  });
});
