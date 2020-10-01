$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        var model = $('#model option:selected').text();
        var dataset = $('#disease').val();
        var request_payload ={
          "image":form_data,
          "model":model,
          "dataset":dataset
        }

        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                var result = data
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html('<h4>Result:</h4>' + '<b>Message: </b>' +result.message + '<br>' + '<b>Probability of Infection: </b>' +result.infection + '<br>' + '<b>Probability of  No Infection: </b>' +result.noinfection);
                console.log('Success!');
            },
        });
    });

});
