document.querySelector('form').addEventListener("submit", form_handler);
        function form_handler(event){
            event.preventDefault();
            send_data();
         }

        function send_data(){
            var fd = new FormData(document.querySelector('form'));
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/predict', true);
            document.getElementById("prediction").innerHTML = "Wait Predicting disease!...";
            xhr.onreadystatechange = function(){
                if(xhr.readyState == XMLHttpRequest.DONE){
                    document.getElementById('prediction').innerHTML="Prediction : " +xhr.responseText;
                }
            };
            xhr.onload = function(){};
            xhr.send(fd);
        }